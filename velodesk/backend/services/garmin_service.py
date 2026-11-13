"""
Garmin Connect integration using python-garminconnect >= 0.3.2
Uses client.connectapi() directly with plain dicts — no pydantic dependency.
Tokens stored in /config/strava_training/garmin_tokens (HA config volume).
"""
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("/config/strava_training/garmin_tokens")


class GarminService:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self._client = None

    async def connect(self) -> bool:
        try:
            from garminconnect import Garmin

            TOKEN_PATH.mkdir(parents=True, exist_ok=True)
            token_str = str(TOKEN_PATH)
            files = [f.name for f in TOKEN_PATH.iterdir()]
            logger.info("Garmin token path: %s  files: %s", token_str, files)

            if files:
                try:
                    client = Garmin()
                    client.login(token_str)
                    self._client = client
                    logger.info("Resumed Garmin session from cached tokens")
                    return True
                except Exception as e:
                    logger.warning("Cached tokens failed: %s", e)

            logger.info("Attempting fresh Garmin login")
            client = Garmin(email=self.email, password=self.password, prompt_mfa=None)
            client.login(token_str)
            self._client = client
            logger.info("Fresh Garmin login OK, tokens saved")
            return True

        except Exception as e:
            if "429" in str(e):
                logger.error("Garmin rate-limited. Copy tokens from PC to %s", TOKEN_PATH)
            else:
                logger.error("Garmin login failed: %s", e)
            return False

    async def export_workout(self, workout) -> Optional[str]:
        """Upload workout to Garmin Connect using upload_workout (POST)."""
        if not self._client:
            if not await self.connect():
                return None
        try:
            payload = self._build_workout(workout)
            response = self._client.upload_workout(payload)
            workout_id = (response or {}).get("workoutId")
            logger.info("Uploaded '%s' to Garmin (ID: %s)", workout.title, workout_id)
            return str(workout_id) if workout_id else None
        except Exception as e:
            logger.error("Garmin export failed: %s", e)
            self._client = None
            return None

    async def schedule_workout(self, garmin_workout_id: str, workout_date) -> bool:
        if not self._client:
            return False
        try:
            date_str = workout_date.strftime("%Y-%m-%d") if hasattr(workout_date, 'strftime') else str(workout_date)[:10]
            self._client.schedule_workout(garmin_workout_id, date_str)
            return True
        except Exception as e:
            logger.error("Schedule failed: %s", e)
            return False

    def _build_workout(self, workout) -> Dict:
        """Build Garmin workout JSON from TrainIQ PlannedWorkout.
        All IDs from garminconnect/workout.py StepType/ConditionType/TargetType enums.
        """
        # Helper constants
        STEP_TYPE = {"warmup": (1,"warmup"), "cooldown": (2,"cooldown"),
                     "interval": (3,"interval"), "recovery": (4,"recovery")}
        COND_TIME = {"conditionTypeId": 2, "conditionTypeKey": "time",
                     "displayOrder": 2, "displayable": True}
        COND_ITER = {"conditionTypeId": 7, "conditionTypeKey": "iterations",
                     "displayOrder": 7, "displayable": True}
        NO_TARGET = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target",
                     "displayOrder": 1}
        SPORT     = {"sportTypeId": 2, "sportTypeKey": "cycling", "displayOrder": 2}

        def step_type_dict(key):
            sid, skey = STEP_TYPE.get(key, (3, "interval"))
            return {"stepTypeId": sid, "stepTypeKey": skey, "displayOrder": sid}

        def power_target(p_low, p_high):
            if not p_low or not p_high or int(p_low) == 0 or int(p_high) == 0:
                return NO_TARGET
            return {"workoutTargetTypeId": 2, "workoutTargetTypeKey": "power.zone",
                    "displayOrder": 1, "targetValueOne": int(p_low), "targetValueTwo": int(p_high)}

        def make_step(order, type_key, dur_s, target):
            return {
                "type": "ExecutableStepDTO",
                "stepOrder": order,
                "stepType": step_type_dict(type_key),
                "endCondition": COND_TIME,
                "endConditionValue": float(dur_s),
                "targetType": target,
            }

        steps = []
        order = 1

        for iv in (workout.intervals or []):
            itype   = iv.get("type", "work")
            dur_s   = int(iv.get("duration_seconds", 300))
            repeats = int(iv.get("repeats", 1))
            rest_s  = int(iv.get("rest_seconds", 0))
            p_low   = iv.get("power_low")
            p_high  = iv.get("power_high")

            skey   = {"warmup": "warmup", "cooldown": "cooldown",
                      "recovery": "recovery"}.get(itype, "interval")
            target = power_target(p_low, p_high) if p_low and p_high else NO_TARGET

            if repeats > 1:
                sub_steps = iv.get("steps")
                if sub_steps:
                    inner = [
                        make_step(i+1,
                                  "interval" if s.get("power_low") else "recovery",
                                  s["duration_seconds"],
                                  power_target(s["power_low"], s["power_high"])
                                  if s.get("power_low") and s.get("power_high")
                                  else NO_TARGET)
                        for i, s in enumerate(sub_steps)
                    ]
                else:
                    inner = [make_step(1, skey, dur_s, target)]
                if rest_s > 0:
                    r_low, r_high = iv.get("rest_power_low"), iv.get("rest_power_high")
                    rest_target = power_target(r_low, r_high) if r_low and r_high else NO_TARGET
                    inner.append(make_step(len(inner)+1, "recovery", rest_s, rest_target))
                steps.append({
                    "type": "RepeatGroupDTO",
                    "stepOrder": order,
                    "stepType": step_type_dict("interval"),  # required by Garmin
                    "numberOfIterations": repeats,
                    "workoutSteps": inner,
                    "endCondition": COND_ITER,
                    "endConditionValue": float(repeats),
                    "smartRepeat": False,
                })
                order += 1
            else:
                steps.append(make_step(order, skey, dur_s, target))
                order += 1
                if rest_s > 0:
                    r_low, r_high = iv.get("rest_power_low"), iv.get("rest_power_high")
                    rest_target = power_target(r_low, r_high) if r_low and r_high else NO_TARGET
                    steps.append(make_step(order, "recovery", rest_s, rest_target))
                    order += 1

        return {
            "workoutName": workout.title,
            "description": workout.description or "",
            "sportType": SPORT,
            "estimatedDurationInSecs": int((workout.target_duration_minutes or 60) * 60),
            "workoutSegments": [{
                "segmentOrder": 1,
                "sportType": SPORT,
                "workoutSteps": steps,
            }],
        }

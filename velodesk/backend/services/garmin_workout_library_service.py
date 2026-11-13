"""
Garmin workout library sync voor TrainIQ.
Haalt cycling-workouts op uit Garmin Connect, labelt bron (join/trainiq/overig)
en classificeert workout_type op basis van berekende zone_seconds.
Upsert-only: verwijderde Garmin-workouts blijven staan, alleen handmatig te verwijderen.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.database import WorkoutLibrary, WorkoutLibraryIgnored, PlannedWorkout
from .training_science import get_power_zones, resolve_intervals_to_watts, watts_to_pct

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("/config/strava_training/garmin_tokens")


class GarminWorkoutLibraryService:
    def __init__(self, email: str, password: str, db: AsyncSession, ftp: float = 200.0):
        self.email = email
        self.password = password
        self.db = db
        self.ftp = ftp
        self._client = None

    async def _get_client(self):
        if self._client:
            return self._client
        try:
            from garminconnect import Garmin

            TOKEN_PATH.mkdir(parents=True, exist_ok=True)
            if not list(TOKEN_PATH.iterdir()):
                logger.error("No Garmin tokens at %s", TOKEN_PATH)
                return None
            client = Garmin()
            client.login(str(TOKEN_PATH))
            self._client = client
            return client
        except Exception as e:
            logger.error("Workout library auth failed: %s", e)
            return None

    def _fetch_all_cycling_workouts(self, client) -> List[Dict[str, Any]]:
        all_workouts: List[Dict[str, Any]] = []
        start, limit = 0, 100
        while True:
            batch = client.get_workouts(start=start, limit=limit)
            if not batch:
                break
            all_workouts.extend(batch)
            if len(batch) < limit:
                break
            start += limit
        return [
            w for w in all_workouts
            if (w.get("sportType") or {}).get("sportTypeKey") == "cycling"
        ]

    def _map_step_type(self, step: Dict[str, Any]) -> str:
        key = ((step.get("stepType") or {}).get("stepTypeKey") or "interval").lower()
        return key if key in ("warmup", "cooldown", "recovery") else "work"

    def _step_duration(self, step: Dict[str, Any]) -> int:
        try:
            return int(step.get("endConditionValue") or 0)
        except (TypeError, ValueError):
            return 0

    def _step_power(self, step: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        target = step.get("targetType") or {}
        target_key = (target.get("workoutTargetTypeKey") or "")
        if target_key.startswith("power."):
            low, high = step.get("targetValueOne"), step.get("targetValueTwo")
            return (int(low) if low else None, int(high) if high else None)
        return (None, None)

    def _parse_garmin_steps(self, workout_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        intervals: List[Dict[str, Any]] = []
        for step in workout_steps or []:
            if step.get("type") == "RepeatGroupDTO":
                inner = step.get("workoutSteps") or []
                repeats = int(step.get("numberOfIterations", 1))
                if len(inner) == 2:
                    work_step, rest_step = inner[0], inner[1]
                    p_low, p_high = self._step_power(work_step)
                    intervals.append({
                        "type": self._map_step_type(work_step),
                        "duration_seconds": self._step_duration(work_step),
                        "repeats": repeats,
                        "rest_seconds": self._step_duration(rest_step),
                        "rest_power_low": self._step_power(rest_step)[0],
                        "rest_power_high": self._step_power(rest_step)[1],
                        "power_low": p_low,
                        "power_high": p_high,
                    })
                elif inner:
                    sub_steps = [
                        {
                            "duration_seconds": self._step_duration(s),
                            "power_low": self._step_power(s)[0],
                            "power_high": self._step_power(s)[1],
                        }
                        for s in inner
                    ]
                    intervals.append({
                        "type": "work",
                        "duration_seconds": sum(s["duration_seconds"] for s in sub_steps),
                        "repeats": repeats,
                        "rest_seconds": 0,
                        "power_low": sub_steps[0]["power_low"],
                        "power_high": sub_steps[0]["power_high"],
                        "steps": sub_steps,
                    })
            elif step.get("type") == "ExecutableStepDTO":
                p_low, p_high = self._step_power(step)
                intervals.append({
                    "type": self._map_step_type(step),
                    "duration_seconds": self._step_duration(step),
                    "repeats": 1,
                    "rest_seconds": 0,
                    "power_low": p_low,
                    "power_high": p_high,
                })
        return intervals

    def _normalize_intervals(
        self, intervals: List[Dict[str, Any]], default_rest_pct: float = 0.55
    ) -> List[Dict[str, Any]]:
        ftp = self.ftp
        warmup_pct: Optional[float] = None
        for iv in intervals:
            if iv.get("type") == "warmup":
                low = watts_to_pct(iv.get("power_low"), ftp)
                high = watts_to_pct(iv.get("power_high"), ftp)
                if low and high:
                    warmup_pct = (low + high) / 2
                    break
        rest_pct: float = warmup_pct or default_rest_pct

        result: List[Dict[str, Any]] = []
        for iv in intervals:
            item: Dict[str, Any] = dict(iv)
            item["ftp_pct_low"] = watts_to_pct(iv.get("power_low"), ftp)
            item["ftp_pct_high"] = watts_to_pct(iv.get("power_high"), ftp)

            steps: List[Dict[str, Any]] = []
            for step in iv.get("steps") or []:
                s: Dict[str, Any] = dict(step)
                s_low = watts_to_pct(step.get("power_low"), ftp)
                s_high = watts_to_pct(step.get("power_high"), ftp)
                if not s_low:
                    s_low = s_high = rest_pct
                s["ftp_pct_low"], s["ftp_pct_high"] = s_low, s_high
                steps.append(s)
            item["steps"] = steps

            if int(iv.get("rest_seconds") or 0) > 0:
                own_low = watts_to_pct(iv.get("rest_power_low"), ftp)
                own_high = watts_to_pct(iv.get("rest_power_high"), ftp)
                item["rest_pct_low"] = own_low or rest_pct
                item["rest_pct_high"] = own_high or own_low or rest_pct

            if iv.get("type") == "recovery" and not item["ftp_pct_low"]:
                item["ftp_pct_low"] = item["ftp_pct_high"] = rest_pct

            result.append(item)
        return result

    def _intervals_for_entry(
        self,
        entry: Optional[WorkoutLibrary],
        parsed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if entry and entry.intervals and entry.raw_garmin_json:
            old_segments = (entry.raw_garmin_json or {}).get("workoutSegments") or []
            old_steps = old_segments[0].get("workoutSteps") if old_segments else []
            has_pct = any(iv.get("ftp_pct_low") for iv in entry.intervals)
            if has_pct and self._parse_garmin_steps(old_steps) == parsed:
                return entry.intervals
        return self._normalize_intervals(parsed)

    def _compute_zone_seconds(self, intervals: List[Dict[str, Any]], ftp: float) -> Dict[str, int]:
        zones = get_power_zones(ftp)
        zone_seconds = {f"Z{z['zone']}": 0 for z in zones}
        for iv in intervals:
            p_low, p_high = iv.get("power_low"), iv.get("power_high")
            dur = int(iv.get("duration_seconds", 0)) * int(iv.get("repeats", 1))
            if not p_low or not p_high or dur <= 0:
                continue
            avg_p = (p_low + p_high) / 2
            for z in zones:
                if z["min"] <= avg_p <= z["max"]:
                    zone_seconds[f"Z{z['zone']}"] += dur
                    break
        return zone_seconds

    def _classify_workout_type(self, zone_seconds: Dict[str, int]) -> str:
        total = sum(zone_seconds.values())
        if total <= 0:
            return "unclassified"
        high_intensity = zone_seconds.get("Z5", 0) + zone_seconds.get("Z6", 0) + zone_seconds.get("Z7", 0)
        if high_intensity >= 60:
            return "vo2max"
        if zone_seconds.get("Z4", 0) >= 180:
            return "threshold"
        if total < 50 * 60 and (zone_seconds.get("Z1", 0) / total) >= 0.7:
            return "recovery"
        if (zone_seconds.get("Z1", 0) + zone_seconds.get("Z2", 0)) / total >= 0.7:
            return "endurance"
        return "unclassified"

    async def sync_library(self) -> Dict[str, Any]:
        client = await self._get_client()
        if not client:
            return {"error": "Not authenticated"}

        cycling_workouts = self._fetch_all_cycling_workouts(client)

        trainiq_result = await self.db.execute(
            select(PlannedWorkout.garmin_workout_id).where(PlannedWorkout.garmin_workout_id.isnot(None))
        )
        trainiq_ids = {row[0] for row in trainiq_result.all()}

        ignored_result = await self.db.execute(select(WorkoutLibraryIgnored.garmin_workout_id))
        ignored_ids = {row[0] for row in ignored_result.all()}

        created, updated, skipped = 0, 0, 0
        for summary in cycling_workouts:
            workout_id = summary.get("workoutId")
            if not workout_id or str(workout_id) in ignored_ids:
                skipped += 1
                continue
            try:
                detail = client.get_workout_by_id(str(workout_id))
            except Exception as e:
                logger.warning("Could not fetch workout detail %s: %s", workout_id, e)
                skipped += 1
                continue
            if not detail:
                skipped += 1
                continue

            name = detail.get("workoutName") or summary.get("workoutName") or "Untitled"
            segments = detail.get("workoutSegments") or []
            raw_steps = segments[0].get("workoutSteps") if segments else []
            parsed = self._parse_garmin_steps(raw_steps)

            if str(workout_id) in trainiq_ids:
                source = "trainiq"
            elif "join" in name.lower():
                source = "join"
            else:
                source = "overig"

            existing = await self.db.execute(
                select(WorkoutLibrary).where(WorkoutLibrary.garmin_workout_id == str(workout_id))
            )
            entry = existing.scalar_one_or_none()
            estimated_duration_s = int(detail.get("estimatedDurationInSecs") or 0)
            intervals = self._intervals_for_entry(entry, parsed)
            resolved = resolve_intervals_to_watts(intervals, self.ftp)
            has_power = any(iv.get("power_low") for iv in resolved)
            workout_type = (
                self._classify_workout_type(self._compute_zone_seconds(resolved, self.ftp))
                if has_power else "unclassified"
            )

            if entry:
                entry.name = name
                if not entry.source_manual:
                    entry.source = source
                if not entry.workout_type_manual:
                    entry.workout_type = workout_type
                entry.intervals = intervals
                entry.raw_garmin_json = detail
                entry.estimated_duration_s = estimated_duration_s
                entry.last_synced_at = datetime.now()
                updated += 1
            else:
                self.db.add(WorkoutLibrary(
                    garmin_workout_id=str(workout_id),
                    name=name,
                    source=source,
                    workout_type=workout_type,
                    sport_type="cycling",
                    estimated_duration_s=estimated_duration_s,
                    intervals=intervals,
                    raw_garmin_json=detail,
                    imported_at=datetime.now(),
                    last_synced_at=datetime.now(),
                    times_used=0,
                ))
                created += 1

        await self.db.commit()
        return {"created": created, "updated": updated, "skipped": skipped, "total_cycling_found": len(cycling_workouts)}
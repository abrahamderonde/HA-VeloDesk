"""
Main FastAPI application for Strava Training Platform.
"""
import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from datetime import date
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from typing import Any, Dict, List, Optional, Set
from fastapi import UploadFile, File
from .models.database import (
    init_db, get_db, Activity, TrainingMetrics, PowerCurve,
    FTPEstimate, PlannedWorkout, TrainingGoal, StravaToken,
    VisitedGemeente, WorkoutLibrary,
)
from .services.garmin_service import GarminService
from .services.garmin_import_service import GarminImportService
from .services.ai_coach import AICoachService
from .services.training_science import (
    calculate_pmc, fit_critical_power, build_power_curve,
    merge_power_curves, POWER_CURVE_DURATIONS, get_power_zones
)
from .services.gemeente_service import GemeenteService
from .services.ride_analysis import DEFAULT_W_PRIME, analyse_ride, compute_quick_stats
from .services.mqtt_service import MqttService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config from HA options file
def load_config():
    options_file = "/data/options.json"
    if os.path.exists(options_file):
        with open(options_file) as f:
            return json.load(f)
    # Dev fallback from env
    return {
        "garmin_email": os.getenv("GARMIN_EMAIL", ""),
        "garmin_password": os.getenv("GARMIN_PASSWORD", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "athlete_weight_kg": int(os.getenv("ATHLETE_WEIGHT_KG", "70")),
        "ftp_initial": int(os.getenv("FTP_INITIAL", "200")),
    }
CONFIG = load_config()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import_legacy_database()
    await init_db()
    
    # Schedule nightly PMC recalculation
    scheduler.add_job(
        nightly_recalculate,
        "cron", hour=2, minute=0,
        id="nightly_recalc"
    )
    # Schedule nightly Garmin activity sync (1am)
    scheduler.add_job(
        nightly_garmin_sync,
        "cron", hour=1, minute=0,
        id="nightly_garmin_sync"
    )
    # Schedule nightly workout export to Garmin (11:45pm — exports tomorrow's workouts)
    scheduler.add_job(
        nightly_garmin_workout_export,
        "cron", hour=23, minute=45,
        id="nightly_garmin_workout_export"
    )
    # Schedule nightly workout library sync (3am — after PMC/activity jobs)
    scheduler.add_job(
        nightly_workout_library_sync,
        "cron", hour=3, minute=0,
        id="nightly_workout_library_sync"
    )
    scheduler.start()
    mqtt_service = MqttService(CONFIG, handle_mqtt_command)
    mqtt_task = asyncio.create_task(mqtt_service.run())
    logger.info("Strava Training Platform started")
    yield
    mqtt_task.cancel()
    try:
        await mqtt_task
    except asyncio.CancelledError:
        pass
    scheduler.shutdown()

app = FastAPI(
    title="Strava Training Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend
frontend_path = "/app/frontend/dist"
if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=f"{frontend_path}/assets"), name="assets")


async def nightly_recalculate():
    """Nightly job: recalculate PMC and update power curve/FTP."""
    from .models.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        await _auto_complete_commutes(db)
        await recalculate_pmc(db)
        await recalculate_power_curve_and_ftp(db)


async def nightly_garmin_workout_export():
    """Nightly job at 23:45: export tomorrow's planned workouts to Garmin."""
    from .models.database import AsyncSessionLocal
    from datetime import date, timedelta
    async with AsyncSessionLocal() as db:
        try:
            tomorrow = date.today() + timedelta(days=1)
            await _export_workouts_for_date(db, tomorrow)
        except Exception as e:
            logger.error("Nightly workout export failed: %s", e)


async def _export_workouts_for_date(db: AsyncSession, target_date):
    """Export all unexported planned workouts for a given date to Garmin."""
    from datetime import date as date_type
    if isinstance(target_date, date_type):
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())
    else:
        start = datetime.combine(target_date.date(), datetime.min.time())
        end = datetime.combine(target_date.date(), datetime.max.time())

    result = await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.date >= start)
        .where(PlannedWorkout.date <= end)
        .where(PlannedWorkout.exported_to_garmin == False)
        .where(PlannedWorkout.workout_type != "commute")
    )
    workouts = result.scalars().all()
    if not workouts:
        return 0

    garmin = GarminService(
        CONFIG.get("garmin_email", ""),
        CONFIG.get("garmin_password", ""),
    )
    exported = 0
    for workout in workouts:
        ids = await garmin.auto_export_workout(workout)
        if ids.get("garmin_workout_id"):
            workout.garmin_workout_id = ids["garmin_workout_id"]
            workout.garmin_schedule_id = ids.get("garmin_schedule_id")
            workout.exported_to_garmin = True
            exported += 1
    if exported:
        await db.commit()
        logger.info("Auto-exported %d workout(s) to Garmin for %s", exported, target_date)
    return exported


async def _recover_missing_power(db: AsyncSession, svc: GarminImportService, days: int) -> int:
    from .services.training_science import calculate_normalized_power
    client = await svc._get_client()
    if not client:
        logger.error("Power recovery: could not authenticate")
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(Activity)
        .where(Activity.strava_id < 0)
        .where(Activity.average_watts.is_(None))
        .where(Activity.start_date >= cutoff)
    )
    activities = result.scalars().all()

    recovered = 0
    for a in activities:
        garmin_id = abs(a.strava_id)
        power_stream = await svc._fetch_power_stream(client, garmin_id)
        if not power_stream:
            continue
        avg_p = sum(power_stream) / len(power_stream)
        np_real = calculate_normalized_power(power_stream)
        np_for_tss = np_real or avg_p
        date_ftp = await get_ftp_at_date(db, a.start_date)
        if_est = np_for_tss / date_ftp if date_ftp else 0
        a.average_watts = avg_p
        a.tss = round(min((a.elapsed_time or 0) * np_for_tss * if_est / (date_ftp * 3600) * 100, 500), 1)
        a.tss_source = "power" if np_real else "power_avg"
        if np_real:
            a.np = round(np_real)
            a.weighted_avg_watts = round(np_real)
            a.power_stream = power_stream
        recovered += 1

    if recovered:
        await db.commit()
        await recalculate_pmc(db)
    return recovered


async def run_garmin_sync(days: int) -> Dict[str, Any]:
    from .models.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        ftp = await get_current_ftp(db)
        svc = GarminImportService(
            CONFIG.get("garmin_email", ""),
            CONFIG.get("garmin_password", ""),
            db,
            ftp=ftp,
        )
        result = await svc.import_recent(days=days)
        if result.get("imported", 0) > 0:
            await recalculate_pmc(db)
        return result


async def handle_mqtt_command(command: str) -> Dict[str, Any]:
    if command == "fetch_latest":
        return await run_garmin_sync(days=2)
    raise ValueError(f"Unknown command: {command}")


async def nightly_garmin_sync() -> None:
    try:
        result = await run_garmin_sync(days=2)
        logger.info("Nightly Garmin sync: %s", result)
    except Exception as e:
        logger.error("Nightly Garmin sync failed: %s", e)

async def nightly_workout_library_sync():
    from .models.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            ftp = await get_current_ftp(db)
            svc = GarminWorkoutLibraryService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            result = await svc.sync_library()
            logger.info("Nightly workout library sync: %s", result)
        except Exception as e:
            logger.error("Nightly workout library sync failed: %s", e)


@app.post("/trainiq/garmin/import-workout-library")
async def garmin_import_workout_library(background_tasks: BackgroundTasks):
    async def _import():
        from .models.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminWorkoutLibraryService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            result = await svc.sync_library()
            logger.info("Manual workout library sync: %s", result)
    background_tasks.add_task(_import)
    return {"status": "Workout library sync started — check logs for progress"}


@app.get("/trainiq/workout-library")
async def list_workout_library(
    source: str = None,
    workout_type: str = None,
    q: str = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(WorkoutLibrary).order_by(
        WorkoutLibrary.imported_at.desc(), WorkoutLibrary.id.desc()
    )
    if source:
        query = query.where(WorkoutLibrary.source == source)
    if workout_type:
        query = query.where(WorkoutLibrary.workout_type == workout_type)
    if q:
        query = query.where(WorkoutLibrary.name.ilike(f"%{q}%"))
    result = await db.execute(query)
    return [
        {
            "id": w.id,
            "garmin_workout_id": w.garmin_workout_id,
            "name": w.name,
            "source": w.source,
            "workout_type": w.workout_type,
            "rating": w.rating,
            "estimated_duration_s": w.estimated_duration_s,
            "intervals": w.intervals,
            "times_used": w.times_used,
            "imported_at": w.imported_at.isoformat() if w.imported_at else None,
            "last_synced_at": w.last_synced_at.isoformat() if w.last_synced_at else None,
        }
        for w in result.scalars().all()
    ]


@app.patch("/trainiq/workout-library/{entry_id}")
async def update_workout_library_entry(entry_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    result = await db.execute(select(WorkoutLibrary).where(WorkoutLibrary.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Workout not found")

    if "source" in data:
        entry.source = data["source"]
        entry.source_manual = True
    if "workout_type" in data:
        entry.workout_type = data["workout_type"]
        entry.workout_type_manual = True
    if "rating" in data:
        rating = data["rating"]
        if rating is not None and not (1 <= int(rating) <= 5):
            raise HTTPException(status_code=400, detail="rating must be between 1 and 5")
        entry.rating = rating

    await db.commit()
    return {"status": "ok", "id": entry.id}


@app.delete("/trainiq/workout-library/{entry_id}")
async def delete_workout_library_entry(entry_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(WorkoutLibrary).where(WorkoutLibrary.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Workout not found")
    await db.delete(entry)
    await db.commit()
    return {"status": "deleted"}

def _library_signature(entry: WorkoutLibrary) -> Optional[str]:
    if not entry.intervals:
        return None
    return json.dumps(entry.intervals, sort_keys=True, default=str)


@app.post("/trainiq/workout-library/dedupe")
async def dedupe_workout_library(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    result = await db.execute(
        select(WorkoutLibrary).order_by(WorkoutLibrary.imported_at.asc(), WorkoutLibrary.id.asc())
    )
    entries: List[WorkoutLibrary] = list(result.scalars().all())

    groups: Dict[str, List[WorkoutLibrary]] = {}
    for entry in entries:
        sig = _library_signature(entry)
        if sig is not None:
            groups.setdefault(sig, []).append(entry)

    removed: List[Dict[str, Any]] = []
    now = datetime.utcnow()
    for group in groups.values():
        if len(group) < 2:
            continue
        keeper, duplicates = group[0], group[1:]
        for dup in duplicates:
            if keeper.rating is None and dup.rating is not None:
                keeper.rating = dup.rating
            db.add(WorkoutLibraryIgnored(
                garmin_workout_id=dup.garmin_workout_id,
                kept_entry_id=keeper.id,
            ))
            removed.append({"name": dup.name, "kept": keeper.name})
            await db.delete(dup)
        keeper.imported_at = now

    await db.commit()
    return {"duplicates_removed": len(removed), "details": removed}

async def _auto_complete_commutes(db: AsyncSession) -> int:
    pending_result = await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.workout_type == "commute")
        .where(PlannedWorkout.completed.is_(None))
        .where(PlannedWorkout.date <= datetime.now())
    )
    pending: list[PlannedWorkout] = list(pending_result.scalars().all())
    if not pending:
        return 0

    ftp: float = await get_current_ftp(db)
    for w in pending:
        day_start: datetime = w.date.replace(hour=0, minute=0, second=0, microsecond=0)
        real_result = await db.execute(
            select(func.coalesce(func.sum(Activity.tss), 0.0))
            .where(Activity.commute == True)
            .where(Activity.synthetic == False)
            .where(Activity.start_date >= day_start)
            .where(Activity.start_date < day_start + timedelta(days=1))
        )
        real_tss: float = float(real_result.scalar() or 0.0)
        planned_tss: float = float(w.target_tss or 0.0)
        duration_min: int = int(w.target_duration_minutes or 0)

        w.completed = True
        w.actual_duration_minutes = duration_min or None
        if real_tss > 0:
            w.actual_tss = round(real_tss, 1)
            continue

        w.actual_tss = planned_tss
        if planned_tss <= 0 or duration_min <= 0:
            continue
        if_val: float = float(w.target_if or 0.65)
        activity = Activity(
            strava_id=None,
            name=w.title,
            sport_type="Ride",
            start_date=w.date,
            elapsed_time=duration_min * 60,
            moving_time=duration_min * 60,
            distance=0,
            average_watts=if_val * ftp,
            weighted_avg_watts=round(if_val * ftp * 1.05),
            np=round(if_val * ftp * 1.05),
            tss=planned_tss,
            has_power=True,
            trainer=False,
            commute=True,
            synthetic=True,
        )
        db.add(activity)
        await db.flush()
        w.actual_activity_id = activity.id

    await db.commit()
    await recalculate_pmc(db)
    logger.info("Auto-completed %d planned commute(s)", len(pending))
    return len(pending)

async def recalculate_pmc(db: AsyncSession):
    """Recalculate PMC (CTL/ATL/TSB) from all activities."""
    result = await db.execute(
        select(Activity.start_date, Activity.tss)
        .where(Activity.tss != None)
        .order_by(Activity.start_date)
    )
    rows = result.all()

    if not rows:
        return

    from collections import defaultdict
    daily_tss = defaultdict(float)
    for row in rows:
        day = row.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_tss[day] += row.tss or 0

    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = max(today_midnight, max(daily_tss.keys()))
    pmc_data = calculate_pmc(dict(daily_tss), end_date=end_date)

    # Upsert into training_metrics
    for entry in pmc_data:
        date = datetime.fromisoformat(entry["date"])
        result = await db.execute(
            select(TrainingMetrics).where(TrainingMetrics.date == date)
        )
        metric = result.scalar_one_or_none()
        if metric:
            metric.ctl = entry["ctl"]
            metric.atl = entry["atl"]
            metric.tsb = entry["tsb"]
            metric.daily_tss = entry["tss"]
        else:
            metric = TrainingMetrics(
                date=date,
                ctl=entry["ctl"],
                atl=entry["atl"],
                tsb=entry["tsb"],
                daily_tss=entry["tss"],
            )
            db.add(metric)

    await db.commit()
    logger.info("PMC recalculated for %d days", len(pmc_data))


async def recalculate_power_curve_and_ftp(db: AsyncSession):
    """Recalculate power curve from last 60 days and fit CP model."""
    cutoff = datetime.now() - timedelta(days=60)
    result = await db.execute(
        select(Activity)
        .where(Activity.has_power == True)
        .where(Activity.start_date >= cutoff)
        .where(Activity.power_stream != None)
    )
    activities_with_power = result.scalars().all()

    if not activities_with_power:
        return

    from .services.training_science import build_power_curve_after_work, DURABILITY_LEVELS_KJ

    curves_by_level: Dict[int, list] = {kj: [] for kj in DURABILITY_LEVELS_KJ}
    for act in activities_with_power:
        if not act.power_stream:
            continue
        for kj in DURABILITY_LEVELS_KJ:
            curve = build_power_curve_after_work(act.power_stream, kj)
            if curve:
                curves_by_level[kj].append(curve)

    if not curves_by_level[0]:
        return

    merged = merge_power_curves(curves_by_level[0])

    for kj, level_curves in curves_by_level.items():
        level_merged = merge_power_curves(level_curves) if level_curves else {}
        rows = (await db.execute(
            select(PowerCurve).where(PowerCurve.work_kj == kj)
        )).scalars().all()
        existing_by_dur = {r.duration_seconds: r for r in rows}
        for dur, power in level_merged.items():
            pc = existing_by_dur.get(dur)
            if pc:
                pc.best_power = power
                pc.updated_at = datetime.now()
            else:
                db.add(PowerCurve(
                    duration_seconds=dur,
                    best_power=power,
                    work_kj=kj,
                    updated_at=datetime.now(),
                ))
        for dur, pc in existing_by_dur.items():
            if dur not in level_merged:
                await db.delete(pc)

    await db.commit()

    # ── CP estimation ────────────────────────────────────────────────────
    # Primary method: single best maximal effort (180s-30min), following
    # intervals.icu's approach. This only needs ONE good effort, not several
    # efforts across a range of durations — much more practical for riders
    # who don't regularly do repeated all-out tests of varying length.
    from .services.training_science import estimate_cp_single_effort

    single_effort_durations = [d for d in merged.keys() if 180 <= d <= 1800]
    cp_result = None
    if single_effort_durations:
        # Try every candidate duration in range and keep whichever yields the
        # HIGHEST estimated CP. Picking by "longest duration" was a bug — a
        # genuine hard 20-min effort should win over a lower-power 30-min
        # tempo/endurance segment that happens to be longer in the merged
        # curve. A real maximal effort produces the highest, most reliable
        # CP estimate, and the single-effort formula is fairly stable across
        # the 3-30min range for a true max effort, so "best CP wins" reliably
        # picks the actual test rather than an incidental longer-duration ride.
        candidates = []
        for dur in single_effort_durations:
            power = merged[dur]
            result = estimate_cp_single_effort(float(dur), float(power))
            candidates.append((result["cp"], dur, power, result))

        best_cp, best_dur, best_power, single_result = max(candidates, key=lambda c: c[0])

        cp_result = {
            "cp": single_result["cp"],
            "w_prime": single_result["w_prime"],
            "p_max": single_result["p_max"],
            "r_squared": None,  # not applicable to single-point estimation
            "method": "single_effort",
        }
        logger.info(
            "CP estimated via single-effort method: %.1fW (from %.0fs effort at %.0fW, "
            "chosen from %d candidates: %s)",
            single_result["cp"], best_dur, best_power, len(candidates),
            {int(d): round(p) for _, d, p, _ in candidates}
        )
    else:
        # Fallback: multi-point curve fit, for cases with no single effort in
        # the 180-1800s range but enough varied-duration data points to fit
        cp_result = fit_critical_power(merged)
        if cp_result:
            cp_result["method"] = "multi_point"
            logger.info(
                "CP estimated via multi-point fit (no single 180-1800s effort available): %.1fW",
                cp_result["cp"]
            )

    if cp_result:
        estimate = FTPEstimate(
            estimated_at=datetime.now(),
            cp=cp_result["cp"],
            w_prime=cp_result["w_prime"],
            p_max=cp_result["p_max"],
            r_squared=cp_result["r_squared"],
            data_window_days=60,
            estimation_method=cp_result.get("method", "multi_point"),
        )
        db.add(estimate)
        await db.commit()
        logger.info(
            "FTP estimated: CP=%.1fW, W'=%.0fJ, method=%s, R²=%s",
            cp_result["cp"], cp_result["w_prime"], cp_result.get("method", "multi_point"),
            f"{cp_result['r_squared']:.3f}" if cp_result.get("r_squared") is not None else "n/a"
        )


# ─── Settings ────────────────────────────────────────────────────────────────

@app.get("/trainiq/settings")
async def get_settings() -> dict:
    return {
        "athlete_weight_kg": CONFIG.get("athlete_weight_kg", 70),
        "ftp_initial": CONFIG.get("ftp_initial", 200),
        "garmin_configured": bool(CONFIG.get("garmin_email")),
        "anthropic_configured": bool(CONFIG.get("anthropic_api_key")),
    }


# ─── Garmin Import ────────────────────────────────────────────────────────────

@app.get("/trainiq/garmin/status")
async def garmin_import_status(db: AsyncSession = Depends(get_db)):
    """Check if Garmin tokens are available for activity import."""
    svc = GarminImportService(
        CONFIG.get("garmin_email", ""),
        CONFIG.get("garmin_password", ""),
        db,
    )
    return await svc.get_auth_status()


@app.get("/trainiq/garmin/cleanup-power-streams")
@app.post("/trainiq/garmin/cleanup-power-streams")
async def cleanup_garmin_power_streams(db: AsyncSession = Depends(get_db)):
    """One-time cleanup: wipe power_stream on all Garmin-imported activities
    (strava_id < 0) where the stream data was garbage from get_activity_details."""
    from sqlalchemy import update
    result = await db.execute(
        update(Activity)
        .where(Activity.strava_id < 0)
        .where(Activity.power_stream.isnot(None))
        .values(power_stream=None)
    )
    await db.commit()
    # Rebuild power curve without the bad streams
    await recalculate_power_curve_and_ftp(db)
    return {"status": "ok", "cleaned": result.rowcount}



@app.post("/trainiq/garmin/backfill-latlng")
async def garmin_backfill_latlng(background_tasks: BackgroundTasks):
    """Backfill GPS tracks for previously-imported Garmin activities that are missing them.
    Needed for gemeente/NL Challenge detection on rides imported before GPS support was added."""
    async def _backfill():
        from .models.database import AsyncSessionLocal
        from .services.garmin_import_service import GarminImportService

        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            client = await svc._get_client()
            if not client:
                logger.error("Backfill latlng: could not authenticate")
                return

            result = await db.execute(
                select(Activity)
                .where(Activity.strava_id < 0)
                .where(Activity.latlng_stream.is_(None))
                .where(Activity.distance > 500)
            )
            activities = result.scalars().all()
            total = len(activities)
            logger.info("Backfilling GPS for %d Garmin activities", total)

            updated = 0
            for i, a in enumerate(activities):
                try:
                    garmin_id = abs(a.strava_id)
                    stream = await svc._fetch_latlng_stream(client, garmin_id)
                    if stream:
                        a.latlng_stream = stream
                        updated += 1
                except Exception as e:
                    logger.warning("Skipping activity %s during GPS backfill: %s", a.id, e)
                    continue

                if (i + 1) % 25 == 0:
                    await db.commit()
                    logger.info("GPS backfill progress: %d/%d (updated=%d)", i + 1, total, updated)

            await db.commit()
            logger.info("GPS backfill complete: %d/%d activities updated", updated, total)

            try:
                await _scan_all_gemeenten()
            except Exception as e:
                logger.warning("Gemeente rescan after backfill failed: %s", e)

    background_tasks.add_task(_backfill)
    return {"status": "Backfill started — check logs for progress"}


@app.post("/trainiq/garmin/backfill-missing-power")
async def garmin_backfill_missing_power(
    background_tasks: BackgroundTasks,
    days: int = 30,
):
    """Re-check recent Garmin-imported activities that have no power data by
    re-downloading their TCX file. Shares logic with the automatic post-sync
    recovery run by nightly_garmin_sync and /trainiq/garmin/import-recent."""
    async def _backfill():
        from .models.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            recovered = await _recover_missing_power(db, svc, days=days)
            logger.info("Power backfill complete: recovered %d activities", recovered)

    background_tasks.add_task(_backfill)
    return {"status": f"Checking last {days} days for missing power — check logs for results"}


@app.post("/trainiq/garmin/recalculate-tss")
async def garmin_recalculate_tss(background_tasks: BackgroundTasks):
    """Recalculate TSS for existing Garmin-imported activities using real Normalized
    Power (fetched via TCX) instead of the previous average-power approximation."""
    async def _recalc():
        from .models.database import AsyncSessionLocal
        from .services.garmin_import_service import GarminImportService
        from .services.training_science import calculate_normalized_power, estimate_tss_from_hr

        async with AsyncSessionLocal() as db:
            current_ftp = await get_current_ftp(db)  # only used for GarminImportService init
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=current_ftp,
            )
            client = await svc._get_client()
            if not client:
                logger.error("Recalculate TSS: could not authenticate")
                return

            result = await db.execute(
                select(Activity).where(Activity.strava_id < 0)
            )
            activities = result.scalars().all()
            total = len(activities)
            logger.info("Recalculating TSS for %d Garmin activities using date-specific FTP", total)

            updated = 0
            for i, a in enumerate(activities):
                try:
                    garmin_id = abs(a.strava_id)
                    old_tss = a.tss
                    if_est = 0.0
                    # Use the FTP that was actually in effect on this activity's date
                    ftp = await get_ftp_at_date(db, a.start_date)

                    power_stream = await svc._fetch_power_stream(client, garmin_id)
                    np_real = calculate_normalized_power(power_stream) if power_stream else None

                    np_for_tss = np_real if np_real and np_real > 0 else a.average_watts
                    if np_for_tss and np_for_tss > 0 and ftp > 0 and a.elapsed_time:
                        if_est = np_for_tss / ftp
                        new_tss = (a.elapsed_time * np_for_tss * if_est) / (ftp * 3600) * 100
                        a.tss = round(min(new_tss, 500), 1)
                        a.tss_source = "power" if np_real else "power_avg"
                        if np_real:
                            a.np = round(np_real)
                            a.weighted_avg_watts = round(np_real)
                            a.power_stream = power_stream
                        updated += 1
                    elif a.average_heartrate and a.elapsed_time:
                        new_tss = estimate_tss_from_hr(
                            a.elapsed_time, a.average_heartrate,
                            a.max_heartrate or CONFIG.get("max_hr", 185),
                            a.sport_type,
                            rest_hr=CONFIG.get("rest_hr", 50),
                        )
                        a.tss = round(min(new_tss, 500), 1)
                        a.tss_source = "hr"
                        updated += 1

                    # Log full breakdown for any activity with high TSS, so root causes
                    # are visible directly in the addon log without a separate debug page
                    if a.tss and a.tss > 150:
                        logger.warning(
                            "HIGH TSS '%s' (%s): TSS=%.0f (was %.0f) | elapsed=%dmin | "
                            "FTP_used=%.0fW | avg_power=%s | NP=%s | IF=%.2f | source=%s",
                            a.name, a.start_date.date().isoformat(), a.tss, old_tss or 0,
                            (a.elapsed_time or 0) // 60, ftp,
                            f"{a.average_watts:.0f}W" if a.average_watts else "none",
                            f"{np_for_tss:.0f}W" if np_for_tss else "none",
                            if_est, a.tss_source
                        )
                except Exception as e:
                    logger.warning("Skipping activity %s during recalc: %s", a.id, e)
                    continue

                # Commit every 25 activities so progress isn't lost on failure/timeout
                if (i + 1) % 25 == 0:
                    await db.commit()
                    logger.info("Recalc progress: %d/%d (updated=%d)", i + 1, total, updated)

            await db.commit()
            logger.info("TSS recalculation complete: %d/%d activities updated", updated, total)
            await recalculate_pmc(db)
            await recalculate_power_curve_and_ftp(db)

    background_tasks.add_task(_recalc)
    return {"status": "TSS recalculation started — check logs for progress"}


@app.post("/trainiq/garmin/import-recent")
async def garmin_import_recent(background_tasks: BackgroundTasks):
    """Import activities from the last 14 days from Garmin Connect, then retry
    power fetch for any activities Garmin hadn't finished processing yet."""
    async def _import():
        from .models.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            result = await svc.import_recent(days=14)
            logger.info("Garmin recent import: %s", result)
            if result.get("imported", 0) > 0:
                await recalculate_pmc(db)
            recovered = await _recover_missing_power(db, svc, days=14)
            if recovered:
                logger.info("Garmin recent import: recovered power for %d activities", recovered)
    background_tasks.add_task(_import)
    return {"status": "Import started", "note": "Check logs for progress"}


@app.post("/trainiq/garmin/import-history")
async def garmin_import_history(
    background_tasks: BackgroundTasks,
    days: int = 365,
):
    """Import full activity history from Garmin Connect."""
    async def _import():
        from .models.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            result = await svc.import_history(days=days)
            logger.info("Garmin history import: %s", result)
            if result.get("imported", 0) > 0:
                await recalculate_pmc(db)
    background_tasks.add_task(_import)
    return {"status": f"History import started for last {days} days"}


# ─── Import ──────────────────────────────────────────────────────────────────

@app.get("/trainiq/debug/day-activities")
async def debug_day_activities(date: str, db: AsyncSession = Depends(get_db)):
    """Debug: show ALL activities stored for a specific date (YYYY-MM-DD),
    including duplicates from both Strava and Garmin imports, to diagnose
    PMC values that seem doubled or otherwise wrong."""
    target_date = datetime.fromisoformat(date)
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= day_start)
        .where(Activity.start_date < day_end)
        .order_by(Activity.start_date)
    )
    acts = result.scalars().all()

    total_tss = sum(a.tss or 0 for a in acts)

    return {
        "date": date,
        "activity_count": len(acts),
        "total_tss_summed": round(total_tss, 1),
        "activities": [
            {
                "id": a.id,
                "strava_id": a.strava_id,
                "name": a.name,
                "start_date": a.start_date.isoformat(),
                "tss": a.tss,
                "tss_source": a.tss_source,
                "synthetic": a.synthetic,
                "source": "garmin" if (a.strava_id or 0) < 0 else "strava" if a.strava_id else "manual/synthetic",
            }
            for a in acts
        ],
    }


@app.get("/trainiq/debug/find-duplicates")
async def find_duplicate_activities(db: AsyncSession = Depends(get_db)):
    """Find activities that likely represent the same real-world ride imported
    twice — once via Strava (positive strava_id) and once via Garmin (negative
    strava_id) — by matching same date + similar duration. These cause TSS to
    be double-counted in the PMC."""
    result = await db.execute(
        select(Activity)
        .where(Activity.synthetic == False)
        .order_by(Activity.start_date)
    )
    acts = result.scalars().all()

    # Group by date
    from collections import defaultdict
    by_day = defaultdict(list)
    for a in acts:
        day = a.start_date.date().isoformat()
        by_day[day].append(a)

    duplicates = []
    for day, day_acts in by_day.items():
        if len(day_acts) < 2:
            continue
        # Check every pair on this day for likely duplication:
        # similar elapsed_time (within 5%) is a strong signal of the same ride
        for i in range(len(day_acts)):
            for j in range(i + 1, len(day_acts)):
                a1, a2 = day_acts[i], day_acts[j]
                t1, t2 = a1.elapsed_time or 0, a2.elapsed_time or 0
                if t1 == 0 or t2 == 0:
                    continue
                ratio = max(t1, t2) / min(t1, t2)
                if ratio < 1.05:  # durations within 5% of each other
                    duplicates.append({
                        "date": day,
                        "activity_1": {"id": a1.id, "strava_id": a1.strava_id, "name": a1.name,
                                       "elapsed_min": t1 // 60, "tss": a1.tss,
                                       "source": "garmin" if (a1.strava_id or 0) < 0 else "strava"},
                        "activity_2": {"id": a2.id, "strava_id": a2.strava_id, "name": a2.name,
                                       "elapsed_min": t2 // 60, "tss": a2.tss,
                                       "source": "garmin" if (a2.strava_id or 0) < 0 else "strava"},
                        "combined_tss": round((a1.tss or 0) + (a2.tss or 0), 1),
                    })

    return {"duplicate_pairs_found": len(duplicates), "duplicates": duplicates}


@app.delete("/trainiq/debug/resolve-duplicate/{keep_id}/{delete_id}")
async def resolve_duplicate(keep_id: int, delete_id: int, db: AsyncSession = Depends(get_db)):
    """Delete one side of a duplicate pair (identified via find-duplicates) and
    rebuild the PMC. keep_id is not modified; delete_id is removed."""
    result = await db.execute(select(Activity).where(Activity.id == delete_id))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    name = activity.name
    await db.delete(activity)
    await db.commit()
    await recalculate_pmc(db)
    return {"status": "deleted", "deleted_id": delete_id, "kept_id": keep_id, "name": name}



@app.get("/trainiq/debug/tss-detail")
async def tss_detail(days: int = 30, db: AsyncSession = Depends(get_db)):
    """Debug: full TSS calculation breakdown per activity for the last N days,
    to diagnose why some activities show unexpectedly high/low TSS."""
    cutoff = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= cutoff)
        .order_by(Activity.start_date.desc())
    )
    acts = result.scalars().all()

    rows = []
    for a in acts:
        elapsed_h = (a.elapsed_time or 0) / 3600
        ftp_at_date = await get_ftp_at_date(db, a.start_date)

        # Reconstruct the IF and expected TSS given stored values, for comparison
        np_used = a.np or a.weighted_avg_watts or a.average_watts
        if_calc = round(np_used / ftp_at_date, 3) if np_used and ftp_at_date else None
        expected_tss = (
            round(elapsed_h * (if_calc ** 2) * 100, 1)
            if if_calc is not None else None
        )

        # Flags to help spot the root cause at a glance
        flags = []
        if a.tss and a.tss > 250:
            flags.append("very_high_tss")
        if if_calc and if_calc > 1.15:
            flags.append("high_if")
        if if_calc and if_calc < 0.3 and a.average_watts:
            flags.append("low_if_with_power")
        if a.average_watts and a.np and a.np < a.average_watts * 0.9:
            flags.append("np_below_avg")  # NP should never be meaningfully below avg power
        if expected_tss and a.tss and abs(expected_tss - a.tss) > 5:
            flags.append("stored_tss_mismatch")  # stored TSS doesn't match recomputation
        if a.tss_source is None and a.tss:
            flags.append("no_source_tag")  # old data predating tss_source tracking
        if not a.average_watts and not a.average_heartrate and a.tss:
            flags.append("tss_without_power_or_hr")  # RPE-based, or stale

        rows.append({
            "id": a.id,
            "date": a.start_date.isoformat(),
            "name": a.name,
            "sport": a.sport_type,
            "elapsed_min": round((a.elapsed_time or 0) / 60),
            "trainer": a.trainer,
            "tss_source": a.tss_source,
            "stored_tss": round(a.tss, 1) if a.tss else None,
            "expected_tss": expected_tss,
            "ftp_used": round(ftp_at_date, 1),
            "average_watts": round(a.average_watts, 1) if a.average_watts else None,
            "np": round(a.np, 1) if a.np else None,
            "weighted_avg_watts": round(a.weighted_avg_watts, 1) if a.weighted_avg_watts else None,
            "if": if_calc,
            "average_heartrate": a.average_heartrate,
            "rpe": a.rpe,
            "flags": flags,
        })

    flagged = [r for r in rows if r["flags"]]
    return {
        "total_activities": len(rows),
        "flagged_count": len(flagged),
        "flagged": flagged,
        "all": rows,
    }



@app.get("/trainiq/debug/tss-stats")
async def tss_stats(db: AsyncSession = Depends(get_db)):
    """Debug: show TSS distribution to diagnose CTL discrepancy."""
    from sqlalchemy import func
    result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= datetime.now() - timedelta(days=90))
        .order_by(Activity.start_date.desc())
    )
    acts = result.scalars().all()
    rows = []
    for a in acts:
        rows.append({
            "date": a.start_date.date().isoformat(),
            "name": a.name,
            "sport": a.sport_type,
            "duration_min": round((a.moving_time or 0) / 60),
            "tss": round(a.tss or 0, 1),
            "has_power": a.has_power,
            "commute": a.commute,
        })
    total_tss_90d = sum(r["tss"] for r in rows)
    avg_daily_tss = round(total_tss_90d / 90, 1)
    return {"avg_daily_tss_90d": avg_daily_tss, "activities": rows}



@app.get("/trainiq/debug/latlng-stats")
async def latlng_stats(db: AsyncSession = Depends(get_db)):
    """Debug: check how many cycling activities have latlng data."""
    from sqlalchemy import func, case
    CYCLING = ["Ride", "VirtualRide", "EBikeRide", "MountainBikeRide", "GravelRide"]
    result = await db.execute(
        select(Activity).where(Activity.sport_type.in_(CYCLING))
    )
    acts = result.scalars().all()
    null_count = sum(1 for a in acts if a.latlng_stream is None)
    empty_count = sum(1 for a in acts if a.latlng_stream is not None and len(a.latlng_stream) == 0)
    has_data = sum(1 for a in acts if a.latlng_stream and len(a.latlng_stream) > 0)
    sample = next((a for a in acts if a.latlng_stream and len(a.latlng_stream) > 0), None)
    return {
        "total_cycling": len(acts),
        "latlng_null": null_count,
        "latlng_empty_list": empty_count,
        "latlng_has_data": has_data,
        "sample_first_point": sample.latlng_stream[0] if sample else None,
    }

@app.get("/trainiq/debug/workout-library-raw/{library_id}")
async def debug_workout_library_raw(library_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(WorkoutLibrary).where(WorkoutLibrary.id == library_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")

    segments = ((entry.raw_garmin_json or {}).get("workoutSegments") or [])
    steps = segments[0].get("workoutSteps") if segments else []

    def summarize(step):
        return {
            "type": step.get("type"),
            "stepType": step.get("stepType"),
            "targetType": step.get("targetType"),
            "targetValueOne": step.get("targetValueOne"),
            "targetValueTwo": step.get("targetValueTwo"),
            "zoneNumber": step.get("zoneNumber"),
        }

    return {
        "name": entry.name,
        "parsed_intervals": entry.intervals,
        "raw_steps_summary": [summarize(s) for s in steps],
    }



@app.post("/trainiq/commutes/preview")
async def preview_synthetic_commutes(request: Request, db: AsyncSession = Depends(get_db)):
    """Preview how many synthetic commute activities would be created."""
    data = await request.json()
    start_date = datetime.fromisoformat(data["start_date"])
    end_date = datetime.fromisoformat(data["end_date"])
    days_of_week = data.get("days_of_week", [0, 1, 2, 3])
    rides_per_day = data.get("rides_per_day", 2)
    duration_minutes = data.get("duration_minutes", 20)
    intensity_factor = data.get("intensity_factor", 0.65)

    ftp = await get_current_ftp(db)
    tss_per_ride = round((duration_minutes / 60) * (intensity_factor ** 2) * 100, 1)
    tss_per_day = tss_per_ride * rides_per_day

    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() in days_of_week:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return {
        "total_days": len(days),
        "total_rides": len(days) * rides_per_day,
        "tss_per_ride": tss_per_ride,
        "tss_per_day": tss_per_day,
        "total_tss": round(len(days) * tss_per_day, 1),
        "ftp_used": round(ftp, 1),
        "sample_days": days[:5],
    }


@app.post("/trainiq/commutes/generate")
async def generate_synthetic_commutes(request: Request, db: AsyncSession = Depends(get_db)):
    """Generate synthetic commute activities for historical backfill."""
    data = await request.json()
    start_date = datetime.fromisoformat(data["start_date"])
    end_date = datetime.fromisoformat(data["end_date"])
    days_of_week = data.get("days_of_week", [0, 1, 2, 3])
    rides_per_day = data.get("rides_per_day", 2)
    duration_minutes = data.get("duration_minutes", 20)
    intensity_factor = data.get("intensity_factor", 0.65)

    ftp = await get_current_ftp(db)
    duration_seconds = duration_minutes * 60
    tss = round((duration_minutes / 60) * (intensity_factor ** 2) * 100, 1)
    distance_m = (duration_minutes / 60) * 15000  # assume 15 km/h

    # Use a large negative synthetic ID base to avoid conflicts
    # Find current min synthetic ID
    result = await db.execute(
        select(Activity.strava_id)
        .where(Activity.synthetic == True)
        .where(Activity.strava_id != None)
        .order_by(Activity.strava_id)
        .limit(1)
    )
    min_existing = result.scalar_one_or_none()
    next_id = min(min_existing or -1, -1) - 1

    created = 0
    current = start_date
    while current <= end_date:
        if current.weekday() in days_of_week:
            for ride_num in range(rides_per_day):
                hour = 8 if ride_num == 0 else 17
                ride_time = current.replace(hour=hour, minute=0, second=0)
                name = "Morning commute (estimated)" if ride_num == 0 else "Afternoon commute (estimated)"
                activity = Activity(
                    strava_id=next_id,  # negative unique ID
                    name=name,
                    sport_type="Ride",
                    start_date=ride_time,
                    elapsed_time=duration_seconds,
                    moving_time=duration_seconds,
                    distance=distance_m,
                    total_elevation_gain=0,
                    average_speed=distance_m / duration_seconds,
                    max_speed=distance_m / duration_seconds,
                    has_power=False,
                    tss=tss,
                    np=None,
                    if_=intensity_factor,
                    commute=True,
                    synthetic=True,
                )
                db.add(activity)
                next_id -= 1
                created += 1
        current += timedelta(days=1)

    await db.commit()
    logger.info("Generated %d synthetic commute activities", created)
    await recalculate_pmc(db)
    logger.info("PMC rebuilt after commute generation")
    total_tss = round(created * tss, 1)
    return {"created": created, "total_tss": total_tss, "tss_per_ride": tss}


@app.delete("/trainiq/commutes/synthetic")
async def delete_synthetic_commutes(db: AsyncSession = Depends(get_db)):
    """Delete all synthetic commute activities and rebuild PMC."""
    result = await db.execute(
        select(Activity).where(Activity.synthetic == True)
    )
    acts = result.scalars().all()
    # Also catch any with NULL strava_id that were created before the fix
    result2 = await db.execute(
        select(Activity).where(Activity.strava_id == None)
    )
    null_acts = result2.scalars().all()
    all_acts = {a.id: a for a in acts + null_acts}
    count = len(all_acts)
    for act in all_acts.values():
        await db.delete(act)
    await db.commit()
    await recalculate_pmc(db)
    return {"deleted": count}


@app.get("/trainiq/commutes/synthetic/count")
async def count_synthetic_commutes(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Activity).where(Activity.synthetic == True))
    return {"count": len(result.scalars().all())}





# ─── Analytics APIs ──────────────────────────────────────────────────────────

@app.post("/trainiq/analytics/recalculate")
async def trigger_recalculate(background_tasks: BackgroundTasks):
    """Manually trigger PMC + power curve + FTP recalculation."""
    async def _recalc():
        from .models.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await recalculate_pmc(session)
            await recalculate_power_curve_and_ftp(session)
            logger.info("Manual recalculation complete")
    background_tasks.add_task(_recalc)
    return {"status": "Recalculation started"}



@app.get("/trainiq/analytics/pmc")
async def get_pmc(days: int = 120, db: AsyncSession = Depends(get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(TrainingMetrics)
        .where(TrainingMetrics.date >= cutoff)
        .order_by(TrainingMetrics.date)
    )
    metrics = result.scalars().all()

    # Fetch top activity per day (highest TSS) for tooltip
    act_result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= cutoff)
        .where(Activity.synthetic == False)
        .where(Activity.tss > 0)
        .order_by(Activity.tss.desc())
    )
    all_acts = act_result.scalars().all()
    # Build day→name map (highest TSS activity per day)
    day_activity = {}
    for a in all_acts:
        day = a.start_date.date().isoformat()
        if day not in day_activity:
            day_activity[day] = a.name

    return [
        {
            "date": m.date.date().isoformat() if hasattr(m.date, 'date') else m.date.isoformat()[:10],
            "ctl": m.ctl,
            "atl": m.atl,
            "tsb": m.tsb,
            "tss": m.daily_tss,
            "activity_name": day_activity.get(
                m.date.date().isoformat() if hasattr(m.date, 'date') else m.date.isoformat()[:10]
            ),
        }
        for m in metrics
    ]


@app.get("/trainiq/analytics/power-curve")
async def get_power_curve(work_kj: int = 0, db: AsyncSession = Depends(get_db)) -> Dict:
    from .services.training_science import _cp3_model, DURABILITY_LEVELS_KJ
    import numpy as np

    result = await db.execute(
        select(PowerCurve)
        .where(PowerCurve.work_kj == work_kj)
        .order_by(PowerCurve.duration_seconds)
    )
    rows = result.scalars().all()
    actual = [{"duration": r.duration_seconds, "power": r.best_power} for r in rows]

    ftp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    ftp_est = ftp_result.scalar_one_or_none()

    ideal = []
    if work_kj == 0 and ftp_est and ftp_est.cp and ftp_est.w_prime and ftp_est.p_max:
        method = getattr(ftp_est, "estimation_method", "multi_point")
        if method == "single_effort":
            durations = [60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800]
        else:
            durations = [15, 20, 30, 45, 60, 90, 120, 180, 240, 300,
                         420, 600, 900, 1200, 1800, 2400, 3600]

        t = np.array(durations, dtype=float)
        p_ideal = np.minimum(_cp3_model(t, ftp_est.cp, ftp_est.w_prime, ftp_est.p_max), 2000.0)
        ideal = [{"duration": d, "power": round(float(p), 1)} for d, p in zip(durations, p_ideal)]

    return {
        "actual": actual,
        "ideal": ideal,
        "work_kj": work_kj,
        "levels": DURABILITY_LEVELS_KJ,
        "cp": ftp_est.cp if ftp_est else None,
        "w_prime": ftp_est.w_prime if ftp_est else None,
        "p_max": ftp_est.p_max if ftp_est else None,
        "r_squared": ftp_est.r_squared if ftp_est else None,
    }


@app.get("/trainiq/debug/garmin-tokens")
async def debug_garmin_tokens():
    """Debug: check what Garmin token files exist on disk."""
    from pathlib import Path
    token_path = Path("/config/strava_training/garmin_tokens")
    if not token_path.exists():
        return {"status": "directory_missing", "path": str(token_path), "files": []}
    files = list(token_path.iterdir())
    file_info = []
    for f in files:
        info = {"name": f.name, "size_bytes": f.stat().st_size}
        if f.suffix == ".json":
            try:
                import json
                data = json.loads(f.read_text())
                # Show keys but not the actual token values
                info["keys"] = list(data.keys()) if isinstance(data, dict) else "not_a_dict"
            except Exception as e:
                info["parse_error"] = str(e)
        file_info.append(info)
    return {
        "status": "ok" if files else "empty",
        "path": str(token_path),
        "files": file_info,
    }

@app.get("/trainiq/debug/power-stats")
async def power_stats(db: AsyncSession = Depends(get_db)):
    """Debug: diagnose power curve and synthetic commute data."""
    cutoff_60 = datetime.now() - timedelta(days=60)

    result = await db.execute(
        select(Activity)
        .where(Activity.sport_type.in_(["Ride", "VirtualRide", "EBikeRide", "GravelRide"]))
        .where(Activity.start_date >= cutoff_60)
        .where(Activity.synthetic == False)
        .order_by(Activity.start_date.desc())
    )
    recent = result.scalars().all()

    pc_result = await db.execute(select(PowerCurve).order_by(PowerCurve.duration_seconds))
    pc_entries = pc_result.scalars().all()

    # Check synthetic commutes
    syn_result = await db.execute(
        select(Activity).where(Activity.synthetic == True).order_by(Activity.start_date.desc())
    )
    syn_acts = syn_result.scalars().all()

    # Check for negative strava_id activities (old format)
    neg_result = await db.execute(
        select(Activity).where(Activity.strava_id < 0)
    )
    neg_acts = neg_result.scalars().all()

    # Check for null strava_id activities
    null_result = await db.execute(
        select(Activity).where(Activity.strava_id == None)
    )
    null_acts = null_result.scalars().all()

    return {
        "rides_last_60d": len(recent),
        "rides_with_has_power_true": sum(1 for a in recent if a.has_power),
        "rides_with_power_stream": sum(1 for a in recent if a.power_stream),
        "power_curve_entries": len(pc_entries),
        "power_curve_sample": [{"duration": p.duration_seconds, "power": p.best_power} for p in pc_entries[:5]],
        "synthetic_count": len(syn_acts),
        "negative_strava_id_count": len(neg_acts),
        "null_strava_id_count": len(null_acts),
        "synthetic_sample": [
            {"date": a.start_date.isoformat(), "name": a.name, "tss": a.tss, "strava_id": a.strava_id, "synthetic": a.synthetic}
            for a in syn_acts[:5]
        ],
        "negative_id_sample": [
            {"date": a.start_date.isoformat(), "name": a.name, "strava_id": a.strava_id, "synthetic": a.synthetic}
            for a in neg_acts[:5]
        ],
    }



@app.get("/trainiq/analytics/pmc-future")
async def get_pmc_future(days: int = 14, db: AsyncSession = Depends(get_db)):
    """Project future CTL/ATL/TSB based on planned workouts.
    Returns both past (last 30 days) and future (next N days) for seamless chart overlay."""
    import math

    CTL_DECAY = math.exp(-1 / 42)
    ATL_DECAY = math.exp(-1 / 7)
    CTL_GAIN  = 1 - CTL_DECAY
    ATL_GAIN  = 1 - ATL_DECAY

    today = datetime.now().date()

    # Get current CTL/ATL from latest TrainingMetrics
    latest_result = await db.execute(
        select(TrainingMetrics)
        .where(TrainingMetrics.date < datetime.combine(today, datetime.min.time()))
        .order_by(TrainingMetrics.date.desc())
        .limit(1)
    )
    latest = latest_result.scalar_one_or_none()
    if not latest:
        return []

    ctl = latest.ctl or 0
    atl = latest.atl or 0

    # Get planned workouts for the next N days
    future_end = datetime.combine(today + timedelta(days=days), datetime.min.time())
    future_start = datetime.combine(today, datetime.min.time())
    pw_result = await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.date >= future_start)
        .where(PlannedWorkout.date < future_end + timedelta(days=1))
        .where(or_(PlannedWorkout.completed.is_(None), PlannedWorkout.completed == True))
    )
    planned = pw_result.scalars().all()

    done_result = await db.execute(
        select(Activity.start_date, Activity.tss, Activity.commute)
        .where(Activity.start_date >= future_start)
        .where(Activity.tss.isnot(None))
    )
    done_tss: Dict[date, float] = {}
    done_commute_tss: Dict[date, float] = {}
    for row in done_result.all():
        d = row.start_date.date()
        done_tss[d] = done_tss.get(d, 0.0) + float(row.tss or 0)
        if row.commute:
            done_commute_tss[d] = done_commute_tss.get(d, 0.0) + float(row.tss or 0)

    planned_tss: Dict[date, float] = {}
    for w in planned:
        if w.completed is True:
            continue
        day = w.date.date()
        value = float(w.target_tss or 0)
        if w.workout_type == "commute":
            value = max(0.0, value - done_commute_tss.get(day, 0.0))
        planned_tss[day] = planned_tss.get(day, 0.0) + value
        
    projection = []
    for i in range(days + 1):
        day = today + timedelta(days=i)
        tss = done_tss.get(day, 0.0) + planned_tss.get(day, 0.0)
        tsb = ctl - atl
        ctl = ctl * CTL_DECAY + tss * CTL_GAIN
        atl = atl * ATL_DECAY + tss * ATL_GAIN
        projection.append({
            "date": day.isoformat(),
            "ctl": round(ctl, 1),
            "atl": round(atl, 1),
            "tsb": round(tsb, 1),
            "tss": round(tss, 1),
            "planned": True,
            "has_workout": tss > 0,
        })

    return projection



@app.get("/trainiq/analytics/pmc-all")
async def get_pmc_all(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingMetrics).order_by(TrainingMetrics.date)
    )
    metrics = result.scalars().all()
    return [
        {
            "date": m.date.isoformat(),
            "ctl": round(m.ctl, 1) if m.ctl else 0,
            "atl": round(m.atl, 1) if m.atl else 0,
            "tsb": round(m.tsb, 1) if m.tsb else 0,
            "tss": round(m.daily_tss, 1) if m.daily_tss else 0,
        }
        for m in metrics
    ]


@app.get("/trainiq/analytics/distance-by-year")
async def get_distance_by_year(
    exclude_commutes: bool = False,
    exclude_indoor: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Return weekly distance per year for multi-year comparison charts."""
    query = select(
        Activity.start_date,
        Activity.distance,
        Activity.sport_type,
        Activity.commute,
        Activity.trainer,
        Activity.synthetic,
    ).where(
        Activity.sport_type.in_(["Ride", "VirtualRide", "EBikeRide", "MountainBikeRide", "GravelRide"])
    ).where(Activity.distance > 0)

    if exclude_commutes:
        query = query.where(Activity.commute == False)
    if exclude_indoor:
        query = query.where(Activity.sport_type != "VirtualRide")
        query = query.where(Activity.trainer == False)

    # Always exclude synthetic
    query = query.where(Activity.synthetic == False)

    result = await db.execute(query.order_by(Activity.start_date))
    rows = result.all()

    # Group by year and ISO week number → {year: {week: km}}
    from collections import defaultdict
    year_week = defaultdict(lambda: defaultdict(float))
    for row in rows:
        dt = row.start_date
        year = dt.year
        # Day of year 1-365
        doy = dt.timetuple().tm_yday
        year_week[year][doy] += (row.distance or 0) / 1000

    # Return as {year: [{doy, km}]}
    years = {}
    for year, doys in sorted(year_week.items()):
        # Aggregate into weekly buckets (week 1-52)
        weekly = defaultdict(float)
        for doy, km in doys.items():
            week = min(52, (doy - 1) // 7 + 1)
            weekly[week] += km
        years[str(year)] = [{"week": w, "km": round(km, 1)} for w, km in sorted(weekly.items())]

    return years



@app.get("/trainiq/debug/cp-fit-detail")
async def cp_fit_detail(db: AsyncSession = Depends(get_db)):
    """Debug: show exactly which power-curve datapoints go into the CP model fit,
    plus which activities contributed power_stream data — useful for comparing
    against intervals.icu or other tools that may use a different data window
    or fitting method and therefore arrive at a different CP value."""
    from .services.training_science import fit_critical_power

    cutoff = datetime.now() - timedelta(days=60)

    # Activities that DO have usable power_stream data in the fit window
    with_stream = await db.execute(
        select(Activity)
        .where(Activity.has_power == True)
        .where(Activity.start_date >= cutoff)
        .where(Activity.power_stream.isnot(None))
        .order_by(Activity.start_date.desc())
    )
    with_stream_acts = with_stream.scalars().all()

    # Activities that have power (avg_watts) but NO stream — these are silently
    # excluded from the CP fit even though they have power data, which can
    # explain a lower/different CP than tools using average-power-only methods
    without_stream = await db.execute(
        select(Activity)
        .where(Activity.has_power == True)
        .where(Activity.start_date >= cutoff)
        .where(Activity.power_stream.is_(None))
        .order_by(Activity.start_date.desc())
    )
    without_stream_acts = without_stream.scalars().all()

    # Current power curve datapoints actually used for the fit
    pc_result = await db.execute(
        select(PowerCurve).where(PowerCurve.work_kj == 0).order_by(PowerCurve.duration_seconds)
    )
    pc_rows = pc_result.scalars().all()
    power_curve = {r.duration_seconds: r.best_power for r in pc_rows}

    fit_points = [
        {"duration_s": d, "power_w": p}
        for d, p in sorted(power_curve.items())
        if 120 <= d <= 1200  # matches fit_critical_power's min/max_duration
    ]

    fit_result = fit_critical_power(power_curve)

    return {
        "fit_window_days": 60,
        "fit_duration_range_seconds": [120, 1200],
        "datapoints_used_in_fit": fit_points,
        "fit_result": fit_result,
        "activities_with_power_stream": [
            {"id": a.id, "name": a.name, "date": a.start_date.date().isoformat(),
             "elapsed_min": (a.elapsed_time or 0) // 60}
            for a in with_stream_acts
        ],
        "activities_missing_power_stream": [
            {"id": a.id, "name": a.name, "date": a.start_date.date().isoformat(),
             "average_watts": a.average_watts,
             "note": "Has avg power but no stored stream — excluded from CP fit"}
            for a in without_stream_acts
        ],
        "explanation": (
            "This app fits CP using only activities with a stored power_stream "
            "(from TCX), restricted to 2-20 minute efforts, over the last 60 days. "
            "It also caps CP at 105% of your best 20-min power if the raw fit exceeds "
            "110% of that value. Tools like intervals.icu may use a different date "
            "window, a different duration range, no such cap, and/or a different "
            "underlying model (e.g. Monod-Scherrer vs this 3-parameter Morton model) "
            "— all of which can produce a different CP from the same underlying rides."
        ),
    }



@app.get("/trainiq/analytics/ftp")
async def get_ftp_estimate(db: AsyncSession = Depends(get_db)):
    cp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    estimate = cp_result.scalar_one_or_none()

    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()
    user_ftp = float(goal.current_ftp) if goal and goal.current_ftp else float(CONFIG.get("ftp_initial", 200))

    if not estimate:
        return {"ftp": user_ftp, "cp": None, "source": "manual", "r_squared": None}
    return {
        "ftp": user_ftp,           # user-controlled, used for TSS/zones
        "cp": estimate.cp,         # auto-calculated, display only
        "w_prime": estimate.w_prime,
        "p_max": estimate.p_max,
        "r_squared": estimate.r_squared,
        "estimated_at": estimate.estimated_at.isoformat(),
        "source": "cp3_model",
        "estimation_method": getattr(estimate, "estimation_method", "multi_point"),
    }


@app.get("/trainiq/analytics/cp-changed")
async def check_cp_changed(db: AsyncSession = Depends(get_db)):
    """Check if CP has changed significantly since the user was last notified.
    Returns the new CP and user FTP if a notification should be shown."""
    cp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    cp_est = cp_result.scalar_one_or_none()
    if not cp_est:
        return {"changed": False}

    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()

    user_ftp = (goal.current_ftp if goal and goal.current_ftp else CONFIG["ftp_initial"])
    last_notified = getattr(goal, 'last_cp_notified', None) if goal else None
    new_cp = round(cp_est.cp)

    logger.info("cp-changed check: new_cp=%s last_notified=%s user_ftp=%s goal_id=%s",
                new_cp, last_notified, user_ftp, goal.id if goal else None)

    # Notify if CP changed by >2W since last notification
    if last_notified is None or abs(new_cp - last_notified) > 2:
        return {
            "changed": True,
            "new_cp": new_cp,
            "user_ftp": round(user_ftp),
            "difference": round(new_cp - user_ftp),
            "estimated_at": cp_est.estimated_at.isoformat(),
        }
    return {"changed": False}


@app.post("/trainiq/analytics/accept-cp-as-ftp")
async def accept_cp_as_ftp(db: AsyncSession = Depends(get_db)):
    """User accepted the new CP as their FTP — creates a dated FTPHistory entry
    effective today, so historical TSS calculations before this date are unaffected."""
    from .models.database import FTPHistory

    cp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    cp_est = cp_result.scalar_one_or_none()
    if not cp_est:
        raise HTTPException(status_code=404, detail="No CP estimate found")

    today = datetime.combine(datetime.now().date(), datetime.min.time())

    # Upsert: if an entry already exists for today, update it rather than duplicate
    existing = await db.execute(select(FTPHistory).where(FTPHistory.date == today))
    entry = existing.scalar_one_or_none()
    if entry:
        entry.ftp = cp_est.cp
        entry.source = "cp_accepted"
    else:
        db.add(FTPHistory(date=today, ftp=cp_est.cp, source="cp_accepted"))

    # Keep legacy field in sync for any code still reading it directly
    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()
    if not goal:
        goal = TrainingGoal()
        db.add(goal)
    goal.current_ftp = cp_est.cp
    goal.last_cp_notified = cp_est.cp

    await db.commit()
    return {"status": "ok", "new_ftp": round(cp_est.cp), "effective_date": today.date().isoformat()}


@app.get("/trainiq/ftp-history")
async def get_ftp_history(db: AsyncSession = Depends(get_db)):
    """List all FTP history entries, most recent first."""
    from .models.database import FTPHistory
    result = await db.execute(select(FTPHistory).order_by(FTPHistory.date.desc()))
    entries = result.scalars().all()
    return [
        {"id": e.id, "date": e.date.date().isoformat(), "ftp": e.ftp,
         "source": e.source, "notes": e.notes}
        for e in entries
    ]


@app.post("/trainiq/ftp-history")
async def add_ftp_history(request: Request, db: AsyncSession = Depends(get_db)):
    """Add or update a manual FTP entry for a specific date."""
    from .models.database import FTPHistory
    data = await request.json()
    date_str = data.get("date")
    ftp = data.get("ftp")
    notes = data.get("notes")
    if not date_str or not ftp:
        raise HTTPException(status_code=400, detail="date and ftp are required")

    entry_date = datetime.combine(datetime.fromisoformat(date_str).date(), datetime.min.time())
    existing = await db.execute(select(FTPHistory).where(FTPHistory.date == entry_date))
    entry = existing.scalar_one_or_none()
    if entry:
        entry.ftp = float(ftp)
        entry.source = "manual"
        entry.notes = notes
    else:
        entry = FTPHistory(date=entry_date, ftp=float(ftp), source="manual", notes=notes)
        db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return {"status": "ok", "id": entry.id, "date": entry.date.date().isoformat(), "ftp": entry.ftp}


@app.delete("/trainiq/ftp-history/{entry_id}")
async def delete_ftp_history(entry_id: int, db: AsyncSession = Depends(get_db)):
    from .models.database import FTPHistory
    result = await db.execute(select(FTPHistory).where(FTPHistory.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    await db.delete(entry)
    await db.commit()
    return {"status": "deleted"}


@app.get("/trainiq/weight-history")
async def get_weight_history(db: AsyncSession = Depends(get_db)):
    from .models.database import WeightHistory
    result = await db.execute(select(WeightHistory).order_by(WeightHistory.date.desc()))
    entries = result.scalars().all()
    return [
        {"id": e.id, "date": e.date.date().isoformat(), "weight_kg": e.weight_kg, "source": e.source}
        for e in entries
    ]


@app.post("/trainiq/weight-history")
async def add_weight_history(request: Request, db: AsyncSession = Depends(get_db)):
    from .models.database import WeightHistory
    data = await request.json()
    date_str = data.get("date")
    weight_kg = data.get("weight_kg")
    if not date_str or not weight_kg:
        raise HTTPException(status_code=400, detail="date and weight_kg are required")

    entry_date = datetime.combine(datetime.fromisoformat(date_str).date(), datetime.min.time())
    existing = await db.execute(select(WeightHistory).where(WeightHistory.date == entry_date))
    entry = existing.scalar_one_or_none()
    if entry:
        entry.weight_kg = float(weight_kg)
        entry.source = "manual"
    else:
        entry = WeightHistory(date=entry_date, weight_kg=float(weight_kg), source="manual")
        db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return {"status": "ok", "id": entry.id, "date": entry.date.date().isoformat(), "weight_kg": entry.weight_kg}


@app.delete("/trainiq/weight-history/{entry_id}")
async def delete_weight_history(entry_id: int, db: AsyncSession = Depends(get_db)):
    from .models.database import WeightHistory
    result = await db.execute(select(WeightHistory).where(WeightHistory.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    await db.delete(entry)
    await db.commit()
    return {"status": "deleted"}


@app.post("/trainiq/analytics/dismiss-cp-notification")
async def dismiss_cp_notification(db: AsyncSession = Depends(get_db)):
    """User dismissed the CP notification — mark as seen without changing FTP."""
    cp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    cp_est = cp_result.scalar_one_or_none()
    if not cp_est:
        return {"status": "ok"}

    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()
    if goal:
        goal.last_cp_notified = cp_est.cp
        await db.commit()
        logger.info("Dismissed CP notification: set last_cp_notified=%s on goal_id=%s", cp_est.cp, goal.id)
    else:
        logger.warning("Dismiss: no goal found to update")
    return {"status": "dismissed"}




@app.get("/trainiq/analytics/zones")
async def get_zones(db: AsyncSession = Depends(get_db)):
    ftp = await get_current_ftp(db)
    return {"ftp": ftp, "zones": get_power_zones(ftp)}


# ─── Activities ──────────────────────────────────────────────────────────────

@app.get("/trainiq/activities")
async def get_activities(
    page: int = 1,
    per_page: int = 20,
    sport_type: Optional[str] = None,
    exclude_commutes: bool = False,
    db: AsyncSession = Depends(get_db),
) -> List[Dict]:
    query = select(Activity).order_by(Activity.start_date.desc())
    if sport_type:
        query = query.where(Activity.sport_type == sport_type)
    if exclude_commutes:
        query = query.where(
            or_(Activity.commute == False, Activity.commute.is_(None))
        )
    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    activities = result.scalars().all()
    return [_activity_to_dict(a, include_quick=True) for a in activities]


@app.get("/trainiq/activities/calendar")
async def get_calendar_activities(
    year: int = None,
    month: int = None,
    db: AsyncSession = Depends(get_db),
):
    """Get activities for a specific month for calendar view."""
    now = datetime.now()
    year = year or now.year
    month = month or now.month

    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    await _auto_complete_commutes(db)

    result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= start)
        .where(Activity.start_date < end)
        .order_by(Activity.start_date)
    )
    activities = result.scalars().all()

    # Also get planned workouts for this month

    # Also get planned workouts for this month
    pw_result = await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.date >= start)
        .where(PlannedWorkout.date < end)
    )
    planned = pw_result.scalars().all()

    return {
        "activities": [_activity_to_dict(a, include_quick=True) for a in activities],
        "planned": [
            {
                "id": p.id,
                "date": p.date.isoformat(),
                "title": p.title,
                "workout_type": p.workout_type,
                "target_tss": p.target_tss,
                "target_duration_minutes": p.target_duration_minutes,
                "exported_to_garmin": p.exported_to_garmin,
                "completed": p.completed,
                "actual_tss": p.actual_tss,
                "actual_duration_minutes": p.actual_duration_minutes,
            }
            for p in planned
        ],
    }


# ─── Planning ────────────────────────────────────────────────────────────────

@app.get("/trainiq/goals")
async def get_goals(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingGoal).order_by(TrainingGoal.created_at.desc())
    )
    goals = result.scalars().all()
    return [
        {
            "id": g.id,
            "event_name": g.event_name,
            "event_date": g.event_date.isoformat(),
            "event_distance_km": g.event_distance_km,
            "event_elevation_m": g.event_elevation_m,
            "goal_description": g.goal_description,
            "active": g.active,
            "ai_plan_summary": g.ai_plan_summary,
            "weekly_hours": g.weekly_hours,
            "global_plan": g.global_plan,
            "global_plan_generated_at": g.global_plan_generated_at.isoformat() if g.global_plan_generated_at else None,
            "last_week_settings": g.last_week_settings,
        }
        for g in goals
    ]


@app.post("/trainiq/goals")
async def create_goal(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    ftp = await get_current_ftp(db)
    result = await db.execute(
        select(TrainingMetrics).order_by(TrainingMetrics.date.desc()).limit(1)
    )
    latest_metric = result.scalar_one_or_none()
    current_ctl = latest_metric.ctl if latest_metric else 0

    goal = TrainingGoal(
        created_at=datetime.now(),
        event_name=data["event_name"],
        event_date=datetime.fromisoformat(data["event_date"]),
        event_distance_km=data.get("event_distance_km"),
        event_elevation_m=data.get("event_elevation_m"),
        goal_description=data.get("goal_description", ""),
        current_ftp=ftp,
        current_ctl=current_ctl,
        active=True,
        weekly_hours=data.get("weekly_hours"),
    )
    db.add(goal)
    await db.commit()
    await db.refresh(goal)

    if CONFIG.get("anthropic_api_key"):
        ai = AICoachService(CONFIG["anthropic_api_key"])
        summary = await ai.generate_goal_summary(goal, ftp, current_ctl)
        if summary:
            goal.ai_plan_summary = summary
            await db.commit()

    return {"id": goal.id, "status": "created"}


@app.patch("/trainiq/goals/ftp")
async def update_user_ftp(request: Request, db: AsyncSession = Depends(get_db)):
    """Set FTP effective today. Writes an FTPHistory entry so historical TSS before
    today is unaffected; also updates the legacy TrainingGoal.current_ftp field."""
    from .models.database import FTPHistory

    data = await request.json()
    new_ftp = float(data.get("ftp", 0))
    if new_ftp < 50 or new_ftp > 600:
        raise HTTPException(status_code=400, detail="FTP must be between 50 and 600W")

    today = datetime.combine(datetime.now().date(), datetime.min.time())
    existing = await db.execute(select(FTPHistory).where(FTPHistory.date == today))
    entry = existing.scalar_one_or_none()
    if entry:
        entry.ftp = new_ftp
        entry.source = "manual"
    else:
        db.add(FTPHistory(date=today, ftp=new_ftp, source="manual"))

    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()
    if not goal:
        goal = TrainingGoal(created_at=datetime.now(), active=True)
        db.add(goal)
    goal.current_ftp = new_ftp
    await db.commit()
    return {"status": "ok", "ftp": new_ftp}



@app.post("/trainiq/planning/generate-global-plan")
async def generate_global_plan(request: Request, db: AsyncSession = Depends(get_db)):
    """Generate or regenerate the phased global training plan for the active goal."""
    data = await request.json()
    goal_id = data.get("goal_id")

    if not CONFIG.get("anthropic_api_key"):
        raise HTTPException(status_code=400, detail="Anthropic API key not configured")

    result = await db.execute(
        select(TrainingGoal).where(TrainingGoal.id == goal_id) if goal_id
        else select(TrainingGoal).where(TrainingGoal.active == True).limit(1)
    )
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal found")

    # Update weekly_hours if provided
    if "weekly_hours" in data:
        goal.weekly_hours = data["weekly_hours"]

    ftp = await get_current_ftp(db)
    result = await db.execute(
        select(TrainingMetrics).order_by(TrainingMetrics.date.desc()).limit(1)
    )
    latest = result.scalar_one_or_none()
    ctl = latest.ctl if latest else 0

    # Get recent actual weekly TSS for deviation detection
    cutoff = datetime.now() - timedelta(days=28)
    result = await db.execute(
        select(Activity)
        .where(Activity.start_date >= cutoff)
        .order_by(Activity.start_date)
    )
    recent_acts = result.scalars().all()
    # Group by week
    week_map = {}
    for a in recent_acts:
        week = a.start_date.strftime("%Y-%m-%d")[:8] + "01"  # rough week key
        week_key = (a.start_date - timedelta(days=a.start_date.weekday())).strftime("%Y-%m-%d")
        if week_key not in week_map:
            week_map[week_key] = {"week": week_key, "actual_tss": 0, "actual_hours": 0}
        week_map[week_key]["actual_tss"] += a.tss or 0
        week_map[week_key]["actual_hours"] += (a.moving_time or 0) / 3600

    ai = AICoachService(CONFIG["anthropic_api_key"])
    plan = await ai.generate_global_plan(
        goal=goal,
        weekly_hours=goal.weekly_hours or 8,
        current_ctl=ctl,
        ftp=ftp,
        actual_last_weeks=list(week_map.values()),
    )

    if not plan:
        raise HTTPException(status_code=500, detail="Failed to generate global plan")

    goal.global_plan = plan
    goal.global_plan_generated_at = datetime.now()
    await db.commit()
    return plan


@app.post("/trainiq/planning/generate-week")
async def generate_week(request: Request, db: AsyncSession = Depends(get_db)):
    """Generate AI workout plan for a specific week with per-day settings."""
    data = await request.json()
    week_start = datetime.fromisoformat(data["week_start"])
    goal_id = data.get("goal_id")
    # New format: list of day settings
    day_settings = data.get("day_settings", [])

    if not CONFIG.get("anthropic_api_key"):
        raise HTTPException(status_code=400, detail="Anthropic API key not configured")

    result = await db.execute(
        select(TrainingGoal).where(TrainingGoal.id == goal_id) if goal_id
        else select(TrainingGoal).where(TrainingGoal.active == True).limit(1)
    )
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=400, detail="No active training goal found")

    # Save day settings for next week pre-fill
    goal.last_week_settings = day_settings
    await db.commit()

    result = await db.execute(
        select(TrainingMetrics).order_by(TrainingMetrics.date.desc()).limit(1)
    )
    latest = result.scalar_one_or_none()
    ctl = latest.ctl if latest else 0
    atl = latest.atl if latest else 0
    tsb = latest.tsb if latest else 0

    ftp = await get_current_ftp(db)

    cutoff = datetime.now() - timedelta(days=14)
    result = await db.execute(
        select(Activity).where(Activity.start_date >= cutoff).order_by(Activity.start_date.desc())
    )
    recent = result.scalars().all()
    recent_dicts = [_activity_to_dict(a) for a in recent]

    # Find matching week in global plan
    global_plan_week = None
    if goal.global_plan:
        for phase in goal.global_plan.get("phases", []):
            for week in phase.get("weeks", []):
                if week.get("week_start") == week_start.strftime("%Y-%m-%d"):
                    global_plan_week = week
                    break

    few_shot_examples = await _select_few_shot_workouts(db, day_settings, ftp)

    ai = AICoachService(CONFIG["anthropic_api_key"])
    plan = await ai.generate_weekly_plan(
        goal=goal,
        current_ctl=ctl,
        current_atl=atl,
        current_tsb=tsb,
        ftp=ftp,
        recent_activities=recent_dicts,
        week_start=week_start,
        day_settings=day_settings,
        global_plan_week=global_plan_week,
        few_shot_examples=few_shot_examples,
    )

    if not plan:
        raise HTTPException(status_code=500, detail="Failed to generate plan")

    # Delete existing planned workouts for this week before saving new ones
    week_end = week_start + timedelta(days=7)
    existing = (await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.date >= week_start)
        .where(PlannedWorkout.date < week_end)
    )).scalars().all()

    # Delete old workouts from Garmin before replacing
    garmin_to_clean = [(w.garmin_workout_id, getattr(w, 'garmin_schedule_id', None))
                       for w in existing if w.garmin_workout_id]
    if garmin_to_clean:
        garmin = GarminService(CONFIG.get("garmin_email", ""), CONFIG.get("garmin_password", ""))
        for gid, sid in garmin_to_clean:
            try:
                await garmin.delete_garmin_workout(gid, sid)
            except Exception as e:
                logger.warning("Could not delete Garmin workout %s: %s", gid, e)

    for wo in existing:
        await db.delete(wo)

    saved_ids = []
    for wo in plan.get("workouts", []):
        workout = PlannedWorkout(
            date=datetime.fromisoformat(wo["date"]),
            title=wo["title"],
            description=wo["description"],
            workout_type=wo["workout_type"],
            target_tss=wo.get("target_tss"),
            target_duration_minutes=wo.get("target_duration_minutes"),
            target_if=wo.get("target_if"),
            intervals=wo.get("intervals"),
            goal_id=goal.id,
        )
        db.add(workout)
        await db.flush()
        saved_ids.append(workout.id)

    await db.commit()
    plan["saved_workout_ids"] = saved_ids

    # Auto-export any workouts scheduled for today immediately
    from datetime import date as date_type
    today = date_type.today()
    for wo in plan.get("workouts", []):
        try:
            wo_date = datetime.fromisoformat(wo["date"]).date()
            if wo_date == today:
                logger.info("Auto-exporting today's workout '%s' to Garmin", wo["title"])
                await _export_workouts_for_date(db, today)
                break
        except Exception:
            pass

    return plan

@app.post("/trainiq/planning/commutes")
async def save_planned_commutes(request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    data = await request.json()
    week_start = datetime.fromisoformat(data["week_start"])
    week_end = week_start + timedelta(days=7)
    day_settings: list = data.get("day_settings") or []

    existing = (await db.execute(
        select(PlannedWorkout)
        .where(PlannedWorkout.workout_type == "commute")
        .where(PlannedWorkout.date >= week_start)
        .where(PlannedWorkout.date < week_end)
    )).scalars().all()
    for wo in existing:
        await db.delete(wo)

    created = 0
    for ds in day_settings:
        minutes = int(ds.get("commute_minutes") or 0)
        if minutes <= 0:
            continue
        day = datetime.fromisoformat(ds["date"][:10]).replace(hour=7)
        db.add(PlannedWorkout(
            date=day,
            title="Commute",
            description=f"Planned commute ({minutes} min)",
            workout_type="commute",
            target_tss=round((minutes / 60) * (0.65 ** 2) * 100, 1),
            target_duration_minutes=minutes,
            target_if=0.65,
        ))
        created += 1

    await db.commit()
    return {"status": "ok", "created": created}

@app.get("/trainiq/planning/download-fit/{workout_id}")
async def download_fit(workout_id: int, db: AsyncSession = Depends(get_db)):
    """Download a workout as a Garmin FIT file for manual import."""
    from fastapi.responses import Response
    from .services.fit_writer import generate_workout_fit
    result = await db.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
    workout = result.scalar_one_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    try:
        fit_data = generate_workout_fit(workout)
        filename = f"{workout.date.strftime('%Y-%m-%d')}_{workout.title.replace(' ', '_')[:30]}.fit"
        return Response(
            content=fit_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error("FIT generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"FIT generation failed: {e}")



@app.post("/trainiq/planning/export-to-garmin/{workout_id}")
async def export_to_garmin(workout_id: int, db: AsyncSession = Depends(get_db)):
    """Export a planned workout to Garmin Connect. Replaces existing if already exported."""
    result = await db.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
    workout = result.scalar_one_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")

    if not CONFIG.get("garmin_email") or not CONFIG.get("garmin_password"):
        raise HTTPException(status_code=400, detail="Garmin credentials not configured")

    garmin = GarminService(CONFIG["garmin_email"], CONFIG["garmin_password"])

    # Delete previous version from Garmin if it exists
    if workout.garmin_workout_id:
        await garmin.delete_from_garmin(
            workout.garmin_workout_id,
            getattr(workout, "garmin_schedule_id", None),
        )
        workout.garmin_workout_id = None
        workout.garmin_schedule_id = None
        workout.exported_to_garmin = False

    ids = await garmin.auto_export_workout(workout)
    if ids.get("garmin_workout_id"):
        workout.garmin_workout_id = ids["garmin_workout_id"]
        workout.garmin_schedule_id = ids.get("garmin_schedule_id")
        workout.exported_to_garmin = True
        await db.commit()
        return {"status": "exported", "garmin_id": ids["garmin_workout_id"]}
    else:
        raise HTTPException(status_code=500, detail="Failed to export to Garmin")


@app.get("/trainiq/planning/workouts")
async def get_planned_workouts(
    from_date: str = None,
    to_date: str = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(PlannedWorkout).order_by(PlannedWorkout.date)
    if from_date:
        query = query.where(PlannedWorkout.date >= datetime.fromisoformat(from_date))
    if to_date:
        query = query.where(PlannedWorkout.date <= datetime.fromisoformat(to_date))
    result = await db.execute(query)
    workouts = result.scalars().all()
    return [
        {
            "id": w.id,
            "date": w.date.isoformat(),
            "title": w.title,
            "description": w.description,
            "workout_type": w.workout_type,
            "target_tss": w.target_tss,
            "target_duration_minutes": w.target_duration_minutes,
            "target_if": w.target_if,
            "intervals": w.intervals,
            "exported_to_garmin": w.exported_to_garmin,
            "garmin_workout_id": w.garmin_workout_id,
            "completed": w.completed,
            "actual_tss": w.actual_tss,
            "actual_duration_minutes": w.actual_duration_minutes,
            "actual_activity_id": w.actual_activity_id,
        }
        for w in workouts
    ]


@app.post("/trainiq/planning/workouts/{workout_id}/mark")
async def mark_workout(workout_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    """Mark a planned workout as completed or skipped, optionally with adjusted TSS/duration.
    Creates a synthetic activity when marked done."""
    data = await request.json()
    completed = data.get("completed")
    actual_tss = data.get("actual_tss")
    actual_duration = data.get("actual_duration_minutes")
    logger.info("Mark workout %s: completed=%s tss=%s dur=%s", workout_id, completed, actual_tss, actual_duration)

    result = await db.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
    workout = result.scalar_one_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")

    workout.completed = completed
    workout.actual_tss = actual_tss
    workout.actual_duration_minutes = actual_duration

    if completed:
        # Create synthetic activity from planned workout
        tss = actual_tss or workout.target_tss or 50
        duration = (actual_duration or workout.target_duration_minutes or 60) * 60
        ftp = await get_current_ftp(db)
        # Back-calculate intensity_factor from TSS
        if_val = ((tss * ftp * 3600) / (duration * ftp * 100)) ** 0.5 if duration > 0 else 0.7
        avg_power = if_val * ftp

        # Delete previous synthetic activity if re-marking
        if workout.actual_activity_id:
            old = await db.execute(select(Activity).where(Activity.id == workout.actual_activity_id))
            old_act = old.scalar_one_or_none()
            if old_act:
                await db.delete(old_act)

        activity = Activity(
            strava_id=None,
            name=workout.title,
            sport_type="Ride" if workout.workout_type == "commute" else "VirtualRide",
            start_date=workout.date,
            elapsed_time=int(duration),
            moving_time=int(duration),
            distance=0,
            average_watts=avg_power,
            weighted_avg_watts=round(avg_power * 1.05),
            np=round(avg_power * 1.05),
            tss=tss,
            has_power=True,
            trainer=workout.workout_type != "commute",
            commute=workout.workout_type == "commute",
            synthetic=True,
        )
        db.add(activity)
        await db.flush()
        workout.actual_activity_id = activity.id
        logger.info("Created synthetic activity for workout '%s' TSS=%.0f", workout.title, tss)

    elif not completed and workout.actual_activity_id:
        # Skipped — remove synthetic activity if it exists
        old = await db.execute(select(Activity).where(Activity.id == workout.actual_activity_id))
        old_act = old.scalar_one_or_none()
        if old_act and old_act.synthetic:
            await db.delete(old_act)
        workout.actual_activity_id = None

    await db.commit()
    await recalculate_pmc(db)
    return {"status": "ok", "completed": completed, "workout_id": workout_id}


@app.post("/trainiq/planning/workouts/{workout_id}/unmark")
async def unmark_workout(workout_id: int, db: AsyncSession = Depends(get_db)):
    """Reset a workout back to unplanned state (remove done/skipped mark)."""
    result = await db.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
    workout = result.scalar_one_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")

    if workout.actual_activity_id:
        old = await db.execute(select(Activity).where(Activity.id == workout.actual_activity_id))
        old_act = old.scalar_one_or_none()
        if old_act and old_act.synthetic:
            await db.delete(old_act)

    workout.completed = None
    workout.actual_tss = None
    workout.actual_duration_minutes = None
    workout.actual_activity_id = None
    await db.commit()
    await recalculate_pmc(db)
    return {"status": "ok"}


@app.post("/trainiq/activities/{activity_id}/set-rpe")
async def set_activity_rpe(activity_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    """Set RPE (1-10) on an activity with no power/HR data and estimate TSS from it."""
    from .services.training_science import estimate_tss_from_rpe

    data = await request.json()
    rpe = data.get("rpe")
    if rpe is None or not (1 <= float(rpe) <= 10):
        raise HTTPException(status_code=400, detail="rpe must be between 1 and 10")

    result = await db.execute(select(Activity).where(Activity.id == activity_id))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    activity.rpe = float(rpe)
    new_tss = estimate_tss_from_rpe(activity.elapsed_time or 0, float(rpe))
    activity.tss = new_tss
    activity.tss_source = "rpe"
    await db.commit()
    await recalculate_pmc(db)

    return {"status": "ok", "rpe": rpe, "tss": new_tss}


@app.post("/trainiq/activities/{activity_id}/toggle-commute")
async def toggle_activity_commute(activity_id: int, db: AsyncSession = Depends(get_db)):
    """Toggle the commute label on an activity."""
    result = await db.execute(select(Activity).where(Activity.id == activity_id))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    activity.commute = not bool(activity.commute)
    await db.commit()
    return {"status": "ok", "commute": activity.commute}


@app.delete("/trainiq/activities/{activity_id}")
async def delete_activity(activity_id: int, db: AsyncSession = Depends(get_db)):
    """Delete any activity from the database (use to remove duplicates)."""
    result = await db.execute(select(Activity).where(Activity.id == activity_id))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    await db.delete(activity)
    await db.commit()
    await recalculate_pmc(db)
    return {"status": "deleted", "name": activity.name}


@app.get("/trainiq/planning/ftp-test-due")
async def ftp_test_due(db: AsyncSession = Depends(get_db)):
    """Check if an FTP test is due (every 60 days).
    Returns days since last test and whether test is due."""
    # Look for activities named like FTP tests OR high-IF activities >15min
    result = await db.execute(
        select(Activity)
        .where(Activity.elapsed_time >= 20 * 60)  # at least 20min
        .where(Activity.tss > 0)
        .where(Activity.synthetic == False)
        .order_by(Activity.start_date.desc())
        .limit(200)
    )
    acts = result.scalars().all()

    # Find last FTP test by name or by high IF (>1.05 for 20min = likely test)
    last_test_date = None
    for a in acts:
        is_test = (
            'ftp' in (a.name or '').lower() or
            'test' in (a.name or '').lower() or
            (a.if_ and a.if_ > 1.03 and a.elapsed_time >= 20 * 60)
        )
        if is_test:
            last_test_date = a.start_date
            break

    days_since = None
    if last_test_date:
        days_since = (datetime.now() - last_test_date).days

    due = days_since is None or days_since >= 60
    return {
        "due": due,
        "days_since": days_since,
        "last_test_date": last_test_date.isoformat() if last_test_date else None,
        "days_until_due": max(0, 60 - days_since) if days_since is not None else 0,
    }


@app.post("/trainiq/planning/schedule-ftp-test")
async def schedule_ftp_test(request: Request, db: AsyncSession = Depends(get_db)):
    """Schedule an FTP test workout on a given date."""
    data = await request.json()
    date_str = data.get("date")
    if not date_str:
        raise HTTPException(status_code=400, detail="date required")

    ftp = await get_current_ftp(db)
    workout_date = datetime.fromisoformat(date_str)

    # Build FTP test intervals with actual watt targets
    def watts(pct):
        return round(ftp * pct)

    intervals = [
        {"type": "warmup",   "duration_seconds": 300,  "power_low": watts(0.52), "power_high": watts(0.58)},
        {"type": "work",     "duration_seconds": 240,  "power_low": watts(0.72), "power_high": watts(0.78)},
        {"type": "work",     "duration_seconds": 120,  "power_low": watts(0.82), "power_high": watts(0.88)},
        {"type": "work",     "duration_seconds": 60,   "power_low": watts(0.92), "power_high": watts(0.98)},
        {"type": "work",     "duration_seconds": 60,   "power_low": watts(0.97), "power_high": watts(1.03)},
        {"type": "work",     "duration_seconds": 30,   "power_low": watts(1.07), "power_high": watts(1.13)},
        {"type": "recovery", "duration_seconds": 180,  "power_low": watts(0.52), "power_high": watts(0.58)},
        # 2x neuromuscular spins
        {"type": "work",     "duration_seconds": 6,    "power_low": watts(1.90), "power_high": watts(2.10)},
        {"type": "recovery", "duration_seconds": 54,   "power_low": watts(0.52), "power_high": watts(0.58)},
        {"type": "work",     "duration_seconds": 6,    "power_low": watts(1.90), "power_high": watts(2.10)},
        {"type": "recovery", "duration_seconds": 54,   "power_low": watts(0.52), "power_high": watts(0.58)},
        # Settle
        {"type": "recovery", "duration_seconds": 120,  "power_low": watts(0.52), "power_high": watts(0.58)},
        # Optional extra block (reach optimal start)
        {"type": "recovery", "duration_seconds": 360,  "power_low": watts(0.52), "power_high": watts(0.58)},
        # THE TEST — 20min @ 110% (aim slightly above expected FTP)
        {"type": "work",     "duration_seconds": 1200, "power_low": watts(1.07), "power_high": watts(1.13)},
        # Cooldown
        {"type": "cooldown", "duration_seconds": 600,  "power_low": watts(0.52), "power_high": watts(0.58)},
    ]

    total_s = sum(i["duration_seconds"] for i in intervals)
    target_tss = round((total_s * (ftp * 1.05) * (1.05)) / (ftp * 3600) * 100)

    workout = PlannedWorkout(
        date=workout_date,
        title="FTP Test",
        description=(
            f"Structured FTP test. Warmup: 5min easy → 4min tempo → 2min threshold → "
            f"1min VO2 → 1min FTP → 30sec sprint → 3min easy → 2x 6sec sprint. "
            f"Optional 6min extra block to reach optimal start point. "
            f"TEST: 20min @ {watts(1.07)}-{watts(1.13)}W (110% current FTP). "
            f"Average power over 20min × 0.95 = new FTP estimate. "
            f"Cooldown: 10min easy."
        ),
        workout_type="threshold",
        target_tss=target_tss,
        target_duration_minutes=total_s // 60,
        target_if=1.05,
        intervals=intervals,
    )
    db.add(workout)
    await db.commit()
    await db.refresh(workout)

    # Auto-export to Garmin regardless of how far in the future the date is —
    # scheduling via the weekly planner often sets dates several days out.
    # Wrapped so a Garmin failure doesn't hide the fact the workout WAS saved.
    try:
        garmin = GarminService(
            CONFIG.get("garmin_email", ""),
            CONFIG.get("garmin_password", ""),
        )
        result = await garmin.auto_export_workout(workout)
        if result.get("garmin_workout_id"):
            workout.garmin_workout_id = result["garmin_workout_id"]
            workout.garmin_schedule_id = result.get("garmin_schedule_id")
            workout.exported_to_garmin = True
            await db.commit()
        else:
            logger.warning("FTP test saved (id=%d) but Garmin export returned no workout ID", workout.id)
    except Exception as e:
        logger.error("FTP test saved (id=%d) but Garmin export raised an exception: %s", workout.id, e)

    return {
        "status": "ok",
        "workout_id": workout.id,
        "date": date_str,
        "target_watts": f"{watts(1.07)}-{watts(1.13)}W",
        "exported_to_garmin": workout.exported_to_garmin,
    }



@app.post("/trainiq/planning/workouts/add-manual")
async def add_manual_activity(request: Request, db: AsyncSession = Depends(get_db)):
    """Add an unplanned manual activity directly to the PMC (no planned workout needed)."""
    data = await request.json()
    date_str = data.get("date")
    title = data.get("title", "Manual activity")
    tss = float(data.get("tss", 50))
    duration_min = int(data.get("duration_minutes", 60))
    sport_type = data.get("sport_type", "Ride")
    commute = bool(data.get("commute", False))

    ftp = await get_current_ftp(db)
    duration_s = duration_min * 60
    if_val = ((tss * ftp * 3600) / (duration_s * ftp * 100)) ** 0.5 if duration_s > 0 else 0.7

    activity = Activity(
        strava_id=None,
        name=title,
        sport_type=sport_type,
        start_date=datetime.fromisoformat(date_str) if date_str else datetime.now(),
        elapsed_time=duration_s,
        moving_time=duration_s,
        distance=0,
        average_watts=if_val * ftp,
        tss=tss,
        has_power=True,
        trainer=False,
        commute=commute,
        synthetic=True,
    )
    db.add(activity)
    await db.commit()
    await db.refresh(activity)
    await recalculate_pmc(db)
    return {"status": "ok", "activity_id": activity.id, "tss": tss}


@app.post("/trainiq/planning/workouts/from-library")
async def add_workout_from_library(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    library_id = data.get("library_id")
    date_str = data.get("date")
    goal_id = data.get("goal_id")
    if not library_id or not date_str:
        raise HTTPException(status_code=400, detail="library_id and date are required")

    result = await db.execute(select(WorkoutLibrary).where(WorkoutLibrary.id == library_id))
    lib = result.scalar_one_or_none()
    if not lib:
        raise HTTPException(status_code=404, detail="Library workout not found")

    ftp = await get_current_ftp(db)
    duration_s = lib.estimated_duration_s or sum(
        int(iv.get("duration_seconds", 0)) * int(iv.get("repeats", 1)) for iv in (lib.intervals or [])
    )
    weighted_if_sq = 0.0
    for iv in (lib.intervals or []):
        p_low, p_high = iv.get("power_low"), iv.get("power_high")
        dur = int(iv.get("duration_seconds", 0)) * int(iv.get("repeats", 1))
        if p_low and p_high and dur > 0 and ftp > 0:
            avg_p = (p_low + p_high) / 2
            weighted_if_sq += dur * (avg_p / ftp) ** 2
    target_tss = round((weighted_if_sq / 3600) * 100, 1) if duration_s else None

    workout = PlannedWorkout(
        date=datetime.fromisoformat(date_str),
        title=lib.name,
        description=f"Uit workout-bibliotheek ({lib.source})",
        workout_type=lib.workout_type or "endurance",
        target_tss=target_tss,
        target_duration_minutes=round(duration_s / 60) if duration_s else None,
        intervals=lib.intervals,
        goal_id=goal_id,
    )
    db.add(workout)
    lib.times_used += 1
    await db.commit()
    await db.refresh(workout)
    return {"status": "ok", "workout_id": workout.id}


@app.delete("/trainiq/planning/workouts/{workout_id}")
async def delete_planned_workout(workout_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a planned workout and remove it from Garmin Connect."""
    result = await db.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
    workout = result.scalar_one_or_none()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")

    # Remove from Garmin if exported
    if workout.garmin_workout_id:
        garmin = GarminService(
            CONFIG.get("garmin_email", ""),
            CONFIG.get("garmin_password", ""),
        )
        await garmin.delete_garmin_workout(
            workout.garmin_workout_id,
            getattr(workout, "garmin_schedule_id", None),
        )

    # Remove linked synthetic activity
    if workout.actual_activity_id:
        act = await db.execute(select(Activity).where(Activity.id == workout.actual_activity_id))
        old_act = act.scalar_one_or_none()
        if old_act and old_act.synthetic:
            await db.delete(old_act)

    await db.delete(workout)
    await db.commit()
    return {"status": "deleted"}



@app.delete("/trainiq/planning/workouts/manual/{activity_id}")
async def delete_manual_activity(activity_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a manual synthetic activity."""
    result = await db.execute(select(Activity).where(
        Activity.id == activity_id, Activity.synthetic == True
    ))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found or not synthetic")
    await db.delete(activity)
    await db.commit()
    await recalculate_pmc(db)
    return {"status": "deleted"}




async def _select_few_shot_workouts(db: AsyncSession, day_settings: List[Dict], ftp: float) -> List[Dict]:
    """Selecteert relevante library-workouts als few-shot voorbeelden.
    Hard filter: source != trainiq, duur binnen ±25% van een geplande trainingsdag.
    Tiebreaker: hoogste rating (nooit primair filter, alleen bij gelijke kandidaten)."""
    training_minutes = sorted({d["workout_minutes"] for d in day_settings if d.get("workout_minutes")})
    if not training_minutes:
        return []

    result = await db.execute(
        select(WorkoutLibrary)
        .where(WorkoutLibrary.source != "trainiq")
        .where(WorkoutLibrary.estimated_duration_s.isnot(None))
        .where(WorkoutLibrary.intervals.isnot(None))
    )
    candidates = result.scalars().all()

    selected: Dict[int, WorkoutLibrary] = {}
    for target_min in training_minutes:
        low, high = target_min * 0.75 * 60, target_min * 1.25 * 60
        matches = [
            c for c in candidates
            if c.id not in selected and low <= (c.estimated_duration_s or 0) <= high
        ]
        matches.sort(key=lambda c: ((c.rating or 0), -(c.times_used or 0)), reverse=True)
        for m in matches[:2]:
            selected[m.id] = m
        if len(selected) >= 8:
            break

    return [
        {
            "name": w.name,
            "workout_type": w.workout_type or "unclassified",
            "duration_minutes": round((w.estimated_duration_s or 0) / 60),
            "intervals": w.intervals,
        }
        for w in list(selected.values())[:8]
    ]


async def get_current_ftp(db: AsyncSession) -> float:
    """Returns the most recent FTP for TSS/zone calculations and workout targets.
    Priority: 1) latest FTPHistory entry, 2) legacy TrainingGoal.current_ftp,
    3) auto CP estimate, 4) config initial."""
    from .models.database import FTPHistory

    hist_result = await db.execute(
        select(FTPHistory).order_by(FTPHistory.date.desc()).limit(1)
    )
    hist = hist_result.scalar_one_or_none()
    if hist:
        return float(hist.ftp)

    # Legacy fallback for installs with data predating FTPHistory
    goal_result = await db.execute(select(TrainingGoal).order_by(TrainingGoal.id.desc()).limit(1))
    goal = goal_result.scalar_one_or_none()
    if goal and goal.current_ftp:
        return float(goal.current_ftp)

    cp_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    estimate = cp_result.scalar_one_or_none()
    return estimate.cp if estimate else float(CONFIG.get("ftp_initial", 200))


async def get_ftp_at_date(db: AsyncSession, target_date: datetime) -> float:
    """Returns the FTP that was in effect on a specific date — i.e. the most recent
    FTPHistory entry on or before target_date. Used for accurate historical TSS
    recalculation, since FTP changes over time."""
    from .models.database import FTPHistory

    result = await db.execute(
        select(FTPHistory)
        .where(FTPHistory.date <= target_date)
        .order_by(FTPHistory.date.desc())
        .limit(1)
    )
    hist = result.scalar_one_or_none()
    if hist:
        return float(hist.ftp)

    # No history before this date — use the earliest known FTP if one exists later,
    # otherwise fall back to config initial
    earliest_result = await db.execute(
        select(FTPHistory).order_by(FTPHistory.date.asc()).limit(1)
    )
    earliest = earliest_result.scalar_one_or_none()
    if earliest:
        return float(earliest.ftp)

    return float(CONFIG.get("ftp_initial", 200))


async def get_weight_at_date(db: AsyncSession, target_date: datetime) -> float:
    """Returns bodyweight in effect on a specific date, for W/kg calculations."""
    from .models.database import WeightHistory

    result = await db.execute(
        select(WeightHistory)
        .where(WeightHistory.date <= target_date)
        .order_by(WeightHistory.date.desc())
        .limit(1)
    )
    hist = result.scalar_one_or_none()
    if hist:
        return float(hist.weight_kg)
    return float(CONFIG.get("athlete_weight_kg", 70))


def _activity_to_dict(a: Activity, include_quick: bool = False) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": a.id,
        "strava_id": a.strava_id,
        "name": a.name,
        "sport_type": a.sport_type,
        "start_date": a.start_date.isoformat(),
        "elapsed_time": a.elapsed_time,
        "moving_time": a.moving_time,
        "distance": a.distance,
        "total_elevation_gain": a.total_elevation_gain,
        "average_speed": a.average_speed,
        "average_watts": a.average_watts,
        "weighted_avg_watts": a.weighted_avg_watts,
        "average_heartrate": a.average_heartrate,
        "has_power": a.has_power,
        "commute": a.commute,
        "trainer": a.trainer,
        "synthetic": a.synthetic,
        "rpe": a.rpe,
        "tss_source": a.tss_source,
        "tss": a.tss,
        "np": a.np,
        "if_": a.if_,
        "kilojoules": a.kilojoules,
        "has_streams": bool(a.power_stream),
        "quick": None,
    }
    if include_quick:
        data["quick"] = compute_quick_stats(
            a.power_stream,
            a.hr_stream,
            a.moving_stream,
            a.moving_time,
            a.tss,
            a.average_watts,
            a.kilojoules,
        )
    return data


@app.get("/trainiq/activities/{activity_id}/analysis")
async def get_activity_analysis(activity_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    result = await db.execute(select(Activity).where(Activity.id == activity_id))
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    ftp = await get_ftp_at_date(db, activity.start_date)
    weight = await get_weight_at_date(db, activity.start_date)
    estimate_result = await db.execute(
        select(FTPEstimate).order_by(FTPEstimate.estimated_at.desc()).limit(1)
    )
    estimate = estimate_result.scalar_one_or_none()
    cp = float(estimate.cp) if estimate and estimate.cp else ftp
    w_prime = float(estimate.w_prime) if estimate and estimate.w_prime else DEFAULT_W_PRIME

    analysis = await asyncio.to_thread(
        analyse_ride,
        activity.power_stream,
        activity.hr_stream,
        activity.cadence_stream,
        activity.altitude_stream,
        activity.distance_stream,
        activity.moving_stream,
        ftp,
        cp,
        w_prime,
        weight,
    )
    return {
        "activity": _activity_to_dict(activity, include_quick=True),
        "ftp": round(ftp),
        "weight_kg": weight,
        "analysis": analysis,
    }


@app.post("/trainiq/garmin/backfill-streams")
async def garmin_backfill_streams(background_tasks: BackgroundTasks, days: int = 365) -> Dict[str, str]:
    async def _backfill() -> None:
        from .models.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            ftp = await get_current_ftp(db)
            svc = GarminImportService(
                CONFIG.get("garmin_email", ""),
                CONFIG.get("garmin_password", ""),
                db,
                ftp=ftp,
            )
            client = await svc._get_client()
            if not client:
                logger.error("Backfill streams: could not authenticate")
                return

            cutoff = datetime.now() - timedelta(days=days)
            result = await db.execute(
                select(Activity)
                .where(Activity.strava_id < -100000)
                .where(Activity.synthetic == False)
                .where(Activity.moving_stream.is_(None))
                .where(Activity.start_date >= cutoff)
                .order_by(Activity.start_date.desc())
            )
            activities: List[Activity] = list(result.scalars().all())
            total = len(activities)
            logger.info("Backfilling ride streams for %d Garmin activities", total)

            updated = 0
            for i, a in enumerate(activities):
                streams = await svc._fetch_streams(client, abs(a.strava_id))
                if streams:
                    watts = streams.pop("watts", None)
                    if watts:
                        a.power_stream = watts
                    a.hr_stream = streams.get("hr")
                    a.cadence_stream = streams.get("cadence")
                    a.altitude_stream = streams.get("altitude")
                    a.distance_stream = streams.get("distance")
                    a.moving_stream = streams.get("moving")
                    updated += 1
                else:
                    logger.warning(
                        "No streams for '%s' (%s, garmin id %d)",
                        a.name, a.start_date.date().isoformat(), abs(a.strava_id),
                    )
                await asyncio.sleep(1.5)
                if (i + 1) % 25 == 0:
                    await db.commit()
                    logger.info("Stream backfill progress: %d/%d (updated=%d)", i + 1, total, updated)

            await db.commit()
            logger.info("Stream backfill complete: %d/%d activities updated", updated, total)
            await recalculate_power_curve_and_ftp(db)

    background_tasks.add_task(_backfill)
    return {"status": f"Stream backfill started for last {days} days — check logs for progress"}

@app.get("/trainiq/debug/stream-test")
async def debug_stream_test(activity_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    from garminconnect import Garmin

    result = await db.execute(select(Activity).where(Activity.id == activity_id))
    activity = result.scalar_one_or_none()
    if not activity or not activity.strava_id:
        raise HTTPException(status_code=404, detail="Activity not found")

    svc = GarminImportService(CONFIG.get("garmin_email", ""), CONFIG.get("garmin_password", ""), db)
    client = await svc._get_client()
    if not client:
        return {"error": "not authenticated"}

    garmin_id = abs(activity.strava_id)
    report: Dict[str, Any] = {"garmin_id": garmin_id, "name": activity.name, "date": activity.start_date.isoformat()}
    for label, fmt in (("tcx", Garmin.ActivityDownloadFormat.TCX), ("gpx", Garmin.ActivityDownloadFormat.GPX)):
        try:
            data = await asyncio.to_thread(client.download_activity, str(garmin_id), dl_fmt=fmt)
            report[label] = {"ok": True, "bytes": len(data or b"")}
        except Exception as e:
            report[label] = {"ok": False, "error": str(e)[:200]}
    try:
        summary = await asyncio.to_thread(client.get_activity, str(garmin_id))
        report["summary"] = {"ok": True, "owner": (summary or {}).get("ownerDisplayName"), "name": (summary or {}).get("activityName")}
    except Exception as e:
        report["summary"] = {"ok": False, "error": str(e)[:200]}
    return report


# ─── Gemeenten / Municipalities ───────────────────────────────────────────────

@app.get("/trainiq/gemeenten/boundaries")
async def get_gemeente_boundaries(db: AsyncSession = Depends(get_db)):
    """Return all Dutch gemeente boundaries as GeoJSON (from PDOK cache)."""
    svc = GemeenteService(db)
    await svc.ensure_boundaries_loaded()
    return svc.get_boundaries_geojson()


@app.get("/trainiq/gemeenten/visited")
async def get_visited_gemeenten(db: AsyncSession = Depends(get_db)):
    """Return all uniquely visited gemeenten with first visit date."""
    svc = GemeenteService(db)
    visited = await svc.get_visited_gemeenten()
    stats = await svc.get_stats()
    return {"visited": visited, "stats": stats}


@app.post("/trainiq/gemeenten/scan-all")
async def scan_all_activities_for_gemeenten(background_tasks: BackgroundTasks):
    """Re-scan all cycling activities with GPS to detect visited gemeenten."""
    background_tasks.add_task(_scan_all_gemeenten)
    return {"status": "Scanning started in background"}


async def _scan_all_gemeenten():
    import os
    from .models.database import AsyncSessionLocal
    # Delete cached boundaries so they re-download with correct URL
    cache_path = "/data/strava_training/gemeente_boundaries.json.gz"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        logger.info("Cleared gemeente boundary cache")
    async with AsyncSessionLocal() as db:
        svc = GemeenteService(db)
        await svc.ensure_boundaries_loaded()
        result = await db.execute(
            select(Activity)
            .where(Activity.latlng_stream != None)
            .where(Activity.sport_type.in_(
                ["Ride", "VirtualRide", "EBikeRide", "MountainBikeRide", "GravelRide"]
            ))
        )
        acts = result.scalars().all()
        logger.info("Scanning %d cycling activities for gemeenten", len(acts))
        for i, act in enumerate(acts):
            await svc.process_activity_gemeenten(act)
            if (i + 1) % 50 == 0:
                logger.info("Gemeente scan progress: %d/%d activities", i + 1, len(acts))
        logger.info("Gemeente scan complete")


@app.post("/trainiq/gemeenten/check-gpx")
async def check_gpx(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload GPX → returns crossed gemeenten split into new vs already visited."""
    content = await file.read()
    svc = GemeenteService(db)
    result = await svc.check_gpx_new_gemeenten(content)
    return result


# ─── Eddington Number ─────────────────────────────────────────────────────────

@app.get("/trainiq/eddington")
async def get_eddington(db: AsyncSession = Depends(get_db)):
    """
    Calculate Eddington number for cycling.
    E = largest N such that you have ridden >= N km on at least N days.
    """
    CYCLING_TYPES = ["Ride", "VirtualRide", "EBikeRide", "MountainBikeRide", "GravelRide"]
    result = await db.execute(
        select(Activity.start_date, Activity.distance)
        .where(Activity.sport_type.in_(CYCLING_TYPES))
        .where(Activity.distance > 0)
    )
    rows = result.all()

    if not rows:
        return {"e": 0, "next_e": 1, "rides_needed": 1, "gap_progress": 0, "initial_gap": 1, "total_rides": 0, "histogram": []}

    from collections import defaultdict
    # Sum distance per calendar day (multiple rides same day = combined)
    daily_km: dict = defaultdict(float)
    for row in rows:
        day = row.start_date.strftime("%Y-%m-%d")
        daily_km[day] += (row.distance or 0) / 1000.0

    distances = sorted(daily_km.values(), reverse=True)

    # Compute E: count how many days have >= N km, find largest N where count >= N
    e = 0
    for n in range(1, int(max(distances)) + 2):
        count = sum(1 for d in distances if d >= n)
        if count >= n:
            e = n
        else:
            break

    next_e = e + 1
    count_at_next_e = sum(1 for d in distances if d >= next_e)
    rides_needed = next_e - count_at_next_e

    # Get or create the stable milestone denominator for this next_e target
    from .models.database import EddingtonMilestone
    ms_result = await db.execute(
        select(EddingtonMilestone).where(EddingtonMilestone.next_e == next_e)
    )
    milestone = ms_result.scalar_one_or_none()
    if not milestone and rides_needed > 0:
        milestone = EddingtonMilestone(next_e=next_e, initial_gap=rides_needed)
        db.add(milestone)
        await db.commit()
    if milestone and rides_needed > milestone.initial_gap:
        milestone.initial_gap = rides_needed
        await db.commit()
    initial_gap = milestone.initial_gap if milestone else max(1, rides_needed)
    gap_progress = max(0, initial_gap - rides_needed)

    # Progress histogram: for each distance bucket, how many days
    histogram = []
    for n in range(max(1, e - 20), e + 30):
        count = sum(1 for d in distances if d >= n)
        histogram.append({"km": n, "days": count, "needed": n, "achieved": count >= n})

    return {
        "e": e,
        "next_e": next_e,
        "rides_needed": max(0, rides_needed),
        "gap_progress": gap_progress,
        "initial_gap": initial_gap,
        "total_riding_days": len(distances),
        "total_rides": len(rows),
        "max_day_km": round(max(distances), 1),
        "histogram": histogram,
        "top_days": [round(d, 1) for d in distances[:50]],
    }

@app.patch("/trainiq/eddington/milestone")
async def set_eddington_milestone(
    request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, int | str]:
    from .models.database import EddingtonMilestone

    data = await request.json()
    next_e = int(data["next_e"])
    initial_gap = int(data["initial_gap"])
    if initial_gap < 1:
        raise HTTPException(status_code=400, detail="initial_gap must be >= 1")

    result = await db.execute(
        select(EddingtonMilestone).where(EddingtonMilestone.next_e == next_e)
    )
    milestone = result.scalar_one_or_none()
    if milestone:
        milestone.initial_gap = initial_gap
    else:
        db.add(EddingtonMilestone(next_e=next_e, initial_gap=initial_gap))
    await db.commit()
    return {"status": "ok", "next_e": next_e, "initial_gap": initial_gap}

# ─── Frontend SPA catch-all ───────────────────────────────────────────────────

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str) -> Response:
    root: str = os.path.realpath(frontend_path)
    candidate: str = os.path.realpath(os.path.join(root, full_path))
    if full_path and candidate.startswith(root + os.sep) and os.path.isfile(candidate):
        return FileResponse(candidate)
    index_path = f"{frontend_path}/index.html"
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Frontend not built yet</h1>")

"""Activity Registry — Mirror of app.services.activity_service for content definitions."""

from app.services.activity_service import (
    ACTIVITIES,
    RETIRED_ACTIVITIES_MAP,
    is_retired_activity,
    get_retired_activity_mapping,
    get_all_activities,
    get_activity_by_id,
    get_activities_by_category,
)

__all__ = [
    'ACTIVITIES',
    'RETIRED_ACTIVITIES_MAP',
    'is_retired_activity',
    'get_retired_activity_mapping',
    'get_all_activities',
    'get_activity_by_id',
    'get_activities_by_category',
]

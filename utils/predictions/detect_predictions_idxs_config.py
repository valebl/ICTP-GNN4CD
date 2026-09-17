import numpy as np
from datetime import date

def detect_predictions_idxs_config(args):
    """
    Normalize all test scenarios into a unified structure.
    """

    # Case 1: explicit year list (e.g. "1990_1991_1992")
    if args.test_years:
        years = sorted(int(y) for y in args.test_years.split("_"))
        return {
            "mode": "years_list",
            "years": years
        }

    # Case 2: explicit start/end date
    start = date(
        int(args.test_year_start),
        int(args.test_month_start),
        int(args.test_day_start)
    )
    end = date(
        int(args.test_year_end),
        int(args.test_month_end),
        int(args.test_day_end)
    )

    return {
        "mode": "date_range",
        "start": start,
        "end": end
    }

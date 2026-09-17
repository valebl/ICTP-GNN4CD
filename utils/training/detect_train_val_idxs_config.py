from datetime import date
import random

def detect_train_val_idxs_config(args):
    """
    Normalize all train/val configuration scenarios into a unified structure.
    """

    # Case 1: explicit year lists
    if args.train_years and args.val_years:
        train_years = [int(y) for y in args.train_years]
        val_years   = [int(y) for y in args.val_years]
        return {
            "mode": "years_list",
            "train_years": train_years,
            "val_years": val_years
        }

    # Case 2: random sampling from a year range
    elif args.first_year and args.last_year and args.n_val_years:
        all_years = list(range(int(args.first_year), int(args.last_year) + 1))
        val_years = random.sample(all_years, int(args.n_val_years))
        train_years = [y for y in all_years if y not in val_years]
        return {
            "mode": "random_years",
            "train_years": train_years,
            "val_years": val_years
        }

    # Case 3: explicit date range + validation year
    elif args.train_year_start and args.train_month_start and \
         args.train_day_start and args.train_year_end and \
         args.train_month_end and args.train_day_end and \
         args.validation_year:
        
        train_start = date(
            int(args.train_year_start),
            int(args.train_month_start),
            int(args.train_day_start)
        )
        train_end = date(
            int(args.train_year_end),
            int(args.train_month_end),
            int(args.train_day_end)
        )

        return {
            "mode": "date_range",
            "train_start": train_start,
            "train_end": train_end,
            "validation_year": int(args.validation_year)
        }
    
    else:
        raise ValueError("The provided train-val years configuration is not valid.")

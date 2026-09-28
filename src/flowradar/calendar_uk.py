from datetime import date, timedelta
from dateutil.easter import easter

# first Monday of a month
def _first_monday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(0 - d.weekday()) % 7)

# last Monday of a month
def _last_monday(year: int, month: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=d.weekday())

# roll a weekend date forward to Monday
def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

# bank holidays computed from rules
def bank_holidays(year: int) -> dict[date, str]:
    hols: dict[date, str] = {}
    e = easter(year)

    hols[_next_weekday(date(year, 1, 1))] = "New Year's Day"
    hols[e - timedelta(days=2)] = "Good Friday"
    hols[e + timedelta(days=1)] = "Easter Monday"
    hols[_first_monday(year, 5)] = "Early May bank holiday"
    hols[_last_monday(year, 5)] = "Spring bank holiday"
    hols[_last_monday(year, 8)] = "Summer bank holiday"

    xmas = date(year, 12, 25)
    boxing = date(year, 12, 26)
    if xmas.weekday() == 5:
        # Sat 25: substitutes are Mon 27 (Boxing Day) and Tue 28 (Christmas Day)
        hols[date(year, 12, 27)] = "Boxing Day (substitute)"
        hols[date(year, 12, 28)] = "Christmas Day (substitute)"
    elif xmas.weekday() == 6:
        # Sun 25: Boxing Day is Mon 26, Christmas substitute is Tue 27
        hols[date(year, 12, 26)] = "Boxing Day"
        hols[date(year, 12, 27)] = "Christmas Day (substitute)"
    elif boxing.weekday() == 5:
        # Sat 26: Boxing Day substitute is Mon 28
        hols[xmas] = "Christmas Day"
        hols[date(year, 12, 28)] = "Boxing Day (substitute)"
    else:
        hols[xmas] = "Christmas Day"
        hols[boxing] = "Boxing Day"
    return hols

# merged holiday map across a year range
def holiday_set(start_year: int, end_year: int) -> dict[date, str]:
    out: dict[date, str] = {}
    for y in range(start_year, end_year + 1):
        out.update(bank_holidays(y))
    return out

def is_working_day(d: date, hols: dict[date, str]) -> bool:
    return d.weekday() < 5 and d not in hols

# step back until a working day
def prev_working_day(d: date, hols: dict[date, str]) -> date:
    while not is_working_day(d, hols):
        d -= timedelta(days=1)
    return d

# last calendar day of month, moved back to prior working day if needed
def month_end_payday(year: int, month: int, hols: dict[date, str]) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    return prev_working_day(nxt - timedelta(days=1), hols)

# self assessment payments on account
def tax_dates(year: int) -> dict[date, str]:
    return {
        date(year, 1, 31): "Self assessment balancing payment / 1st POA",
        date(year, 7, 31): "Self assessment 2nd POA",
    }

# simplified quarterly VAT settlement dates (7th of month after quarter end + 1 month)
def vat_quarter_dates(year: int) -> list[date]:
    return [date(year, 2, 7), date(year, 5, 7), date(year, 8, 7), date(year, 11, 7)]
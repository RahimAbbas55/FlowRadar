from datetime import date

from flowradar.calendar_uk import (
    bank_holidays,
    holiday_set,
    is_working_day,
    month_end_payday,
)

def test_known_2024_holidays():
    h = bank_holidays(2024)
    assert date(2024, 1, 1) in h
    assert date(2024, 3, 29) in h  # good friday
    assert date(2024, 4, 1) in h  # easter monday
    assert date(2024, 5, 6) in h  # early may
    assert date(2024, 5, 27) in h  # spring
    assert date(2024, 8, 26) in h  # summer
    assert date(2024, 12, 25) in h
    assert date(2024, 12, 26) in h

def test_new_year_substitute_2022():
    # 1 Jan 2022 was a Saturday, substitute Monday 3 Jan
    assert date(2022, 1, 3) in bank_holidays(2022)

def test_christmas_substitute_2021():
    # 25 Dec 2021 was a Saturday: substitutes Mon 27 and Tue 28
    h = bank_holidays(2021)
    assert date(2021, 12, 27) in h
    assert date(2021, 12, 28) in h

def test_christmas_substitute_2022():
    # 25 Dec 2022 was a Sunday: Boxing Day Mon 26, substitute Tue 27
    h = bank_holidays(2022)
    assert date(2022, 12, 26) in h
    assert date(2022, 12, 27) in h

def test_payday_rolls_back():
    hols = holiday_set(2024, 2024)
    # 31 Aug 2024 is a Saturday, prior working day is Fri 30 Aug
    assert month_end_payday(2024, 8, hols) == date(2024, 8, 30)
    # 31 Jan 2024 is a Wednesday
    assert month_end_payday(2024, 1, hols) == date(2024, 1, 31)

def test_working_day():
    hols = holiday_set(2024, 2024)
    assert not is_working_day(date(2024, 12, 25), hols)
    assert not is_working_day(date(2024, 6, 1), hols)  # saturday
    assert is_working_day(date(2024, 6, 3), hols)
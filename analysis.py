"""
Тестовое задание «Вавиот Азия»: суточное потребление воды и поиск аномалий.

Запуск:   python analysis.py
Нужны:    pandas, numpy, matplotlib
Данные:   data/pokazaniya.csv, data/pribory.csv, data/sobytiya.csv
Выход:    results/*.csv  — таблицы по каждому пункту задания
          results/summary.txt — все цифры, которые вошли в ответы
          charts/*.png   — графики к README
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "results"
CHARTS = BASE / "charts"

# ---------------------------------------------------------------------------
# Пороги — все в одном месте
# ---------------------------------------------------------------------------

# Паспорт АКВА (waviot.ru/download288), табл. 3: «максимальный объём воды за сутки»
DAILY_MAX = {"D15": 37.5, "D20": 62.5}
DAILY_MAX_UNKNOWN = 62.5        # у 19 приборов модель не указана — беру больший предел
COUNTER_CAPACITY = 99999.9999   # ёмкость счётного механизма по тому же паспорту, м³

SPIKE_ABS = 2.0                 # м³/сутки — абсолютный порог всплеска
SPIKE_REL = 10                  # во сколько раз больше «обычного» для прибора
SPIKE_REL_MIN = 0.5             # ...и при этом не меньше 0.5 м³ за сутки
BASE_DAYS = 30                  # «обычный» = медиана предыдущих 30 дней
BASE_MIN_DAYS = 14              # ...если в них есть хотя бы 14 дней с данными
BASE_MIN = 0.02                 # базу меньше 20 л/сутки не использую (делить почти на ноль)

RUN_DAYS = 30                   # длина серии для пунктов 4 и 5
LEAK_FLOOR_SHARE = 0.5          # «строгая» утечка: самый маленький день серии ≥ 50% её медианы
LEAK_MIN_FLOOR = 0.05           # ...и не меньше 50 л/сутки

STEP_MONTHS = 3                 # ступенька: 3 месяца до и 3 месяца после
STEP_FACTOR = 3                 # каждый месяц после в 3+ раза выше (ниже) каждого месяца до
STEP_MIN_DIFF = 0.1             # и разница уровней не меньше 100 л/сутки
MONTH_MIN_DAYS = 15             # месяц учитываю, если в нём ≥ 15 дней с данными

MAGNET_WINDOW = 30              # дней до и после первого «магнита»
MAGNET_DROP = 0.5               # падение = среднее после ≤ 50% среднего до

BIT_STEP = 327.68               # шаг сбоев AQUA2, см. пункт 8

LOG = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    LOG.append(line)


# ---------------------------------------------------------------------------
# Загрузка и суточный расход
# ---------------------------------------------------------------------------

def load():
    opts = dict(sep=";", encoding="utf-8-sig", dtype={"anon_id": str})
    readings = pd.read_csv(DATA / "pokazaniya.csv", parse_dates=["data"], **opts)
    meters = pd.read_csv(DATA / "pribory.csv", **opts)
    events = pd.read_csv(DATA / "sobytiya.csv", parse_dates=["data"], **opts)

    assert not readings.duplicated(["anon_id", "data"]).any(), "два показания на одну дату"
    assert readings.anon_id.isin(meters.anon_id).all()

    meters["dn"] = meters.model.str.extract(r"(D15|D20)", expand=False)
    meters["daily_max"] = meters.dn.map(DAILY_MAX).fillna(DAILY_MAX_UNKNOWN)

    # 34 009 строк событий, но это в основном повторы одной тревоги за день.
    # Оставляю факт «у прибора в этот день было событие такого типа».
    events = events.drop_duplicates()
    readings = readings.sort_values(["anon_id", "data"]).reset_index(drop=True)
    return readings, meters, events


def find_glitches(values):
    """
    Накопительный счётчик не может уменьшаться. Если показание выпрыгнуло
    вверх (или провалилось вниз), а потом ряд вернулся к прежнему уровню,
    сбойным считаю само это показание — это не расход воды.
    Если ряд к прежнему уровню не вернулся (сброс, замена прибора),
    показание принимается как есть.
    """
    n = len(values)
    bad = np.zeros(n, dtype=bool)
    last = values[0]
    for i in range(1, n):
        delta = values[i] - last
        ahead = values[i + 1:i + 61]          # смотрю вперёд до 60 показаний
        if delta > 0.3:
            # выросло — опустится ли потом обратно к прежнему уровню?
            if ((ahead >= last - 0.1) & (ahead <= last + delta / 2)).any():
                bad[i] = True
                continue
        elif delta < -0.01:
            # уменьшилось — вернётся ли к прежнему уровню?
            if (ahead >= last - 0.1).any():
                bad[i] = True
                continue
        last = values[i]
    return bad


def diffs(df):
    g = df.groupby("anon_id")
    gap = (df.data - g.data.shift()).dt.days
    change = (df.pokazanie - g.pokazanie.shift()).round(3)   # округляю до литра
    return gap, change


def add_daily(readings, meters):
    r = readings.merge(meters[["anon_id", "bs", "tip", "dn", "daily_max"]], on="anon_id")
    r["glitch"] = False
    for _, idx in r.groupby("anon_id").groups.items():
        r.loc[idx, "glitch"] = find_glitches(r.loc[idx, "pokazanie"].to_numpy())

    # Сырой расход — ровно как в условии: разница соседних календарных дней.
    # Если между показаниями пропуск, суточного расхода нет (не размазываю по дням).
    r["gap"], r["change"] = diffs(r)
    r["daily"] = r.change.where(r.gap == 1)

    # Чистый расход — то же, но сбойные показания выброшены заранее.
    # Его использую для пунктов 2, 4–7, чтобы сбои связи не выглядели как поведение.
    good = r[~r.glitch].copy()
    good["gap_c"], good["change_c"] = diffs(good)
    r["daily_clean"] = good.change_c.where(good.gap_c == 1)
    r.loc[(r.daily_clean < 0) | (r.daily_clean > r.daily_max), "daily_clean"] = np.nan

    # Соседям явно битого дня тоже не доверяю: там часто «половина» сбоя
    # (за сутки пришло несколько пакетов, часть из них битые).
    broken = (r.daily > r.daily_max) | (r.daily < -0.01)
    g = broken.groupby(r.anon_id)
    near = g.shift(1, fill_value=False) | g.shift(-1, fill_value=False)
    r.loc[near, "daily_clean"] = np.nan
    return r


def to_grid(r, col):
    """Таблица дата × прибор; дни без данных — NaN."""
    days = pd.date_range(r.data.min(), r.data.max())
    return r.pivot(index="data", columns="anon_id", values=col).reindex(days)


def runs(r, condition):
    """Серии подряд идущих календарных дней, на которых выполняется condition."""
    d = r[["anon_id", "data", "daily_clean", "paketov_za_sutki"]].copy()
    d["ok"] = condition
    new_run = (~d.ok) | (d.anon_id != d.anon_id.shift()) | (d.data.diff().dt.days != 1)
    d["run_id"] = new_run.cumsum()
    d = d[d.ok]
    out = d.groupby("run_id").agg(
        anon_id=("anon_id", "first"), start=("data", "min"), end=("data", "max"),
        days=("data", "size"), median_m3=("daily_clean", "median"),
        min_m3=("daily_clean", "min"), total_m3=("daily_clean", "sum"),
        packets=("paketov_za_sutki", "sum"))
    return out.reset_index(drop=True)


def table(df, cols):
    return df[cols].to_string(index=False)


# ---------------------------------------------------------------------------
# 1. Отрицательный расход
# ---------------------------------------------------------------------------

def p1_negative(r, events):
    neg = r[r.change < 0].copy()
    prev_glitch = r.groupby("anon_id").glitch.shift(fill_value=False)
    neg["kind"] = np.select(
        [neg.change > -0.01, prev_glitch[neg.index], neg.glitch],
        ["шум меньше 10 л", "возврат после сбойного скачка вверх", "провал и возврат"],
        default="уровень не вернулся (сброс/замена)")

    resets = events[events.sobytie == "сброс"][["anon_id", "data"]]
    near = neg.reset_index().merge(resets, on="anon_id", suffixes=("", "_ev"))
    near = near[(near.data - near.data_ev).dt.days.abs() <= 1]
    neg["reset_event_near"] = neg.index.isin(near["index"])

    daily = neg[neg.gap == 1]
    say("\n1. ОТРИЦАТЕЛЬНЫЙ РАСХОД  (суточная разница < 0)")
    say(f"   соседние дни: {len(daily)} случаев у {daily.anon_id.nunique()} приборов;"
        f" вместе с разницами через пропуск: {len(neg)} у {neg.anon_id.nunique()}")
    say(daily.groupby("kind").agg(cases=("anon_id", "size"), meters=("anon_id", "nunique"),
                                  reset_event_nearby=("reset_event_near", "sum")).to_string())
    say("   приборов по типам:", daily.groupby("tip").anon_id.nunique().to_dict())
    top = daily.sort_values("change").drop_duplicates("anon_id")
    say(table(top.head(6), ["anon_id", "data", "tip", "pokazanie", "change", "kind"]))
    neg.sort_values("change")[["anon_id", "data", "tip", "pokazanie", "change", "gap",
                               "kind", "reset_event_near"]].to_csv(OUT / "1_negative.csv", index=False)
    return neg


# ---------------------------------------------------------------------------
# 2. Всплески
# ---------------------------------------------------------------------------

def p2_spikes(r):
    daily = to_grid(r, "daily_clean")
    # медиана ПРЕДЫДУЩИХ 30 дней: текущий день в базу не попадает
    base = daily.rolling(BASE_DAYS, min_periods=BASE_MIN_DAYS).median().shift(1)
    absolute = daily > SPIKE_ABS
    relative = (daily > SPIKE_REL * base) & (daily >= SPIKE_REL_MIN) & (base >= BASE_MIN)

    flagged = (absolute | relative).stack()
    idx = flagged[flagged].index
    rows = pd.DataFrame({"daily_m3": daily.stack().reindex(idx),
                         "base_m3": base.stack().reindex(idx),
                         "absolute": absolute.stack().reindex(idx),
                         "relative": relative.stack().reindex(idx)}).reset_index()
    rows = rows.rename(columns={"level_0": "data"})
    rows["times_base"] = (rows.daily_m3 / rows.base_m3).round(1)

    say("\n2. ВСПЛЕСКИ  (по очищенному ряду)")
    say(f"   абсолютный, > {SPIKE_ABS} м³/сутки: {absolute.sum().sum()} дней у {absolute.any().sum()} приборов")
    say(f"   относительный, > {SPIKE_REL}× медианы прошлых {BASE_DAYS} дней и ≥ {SPIKE_REL_MIN} м³: "
        f"{relative.sum().sum()} дней у {relative.any().sum()} приборов")
    say(f"   оба условия в один день: {(absolute & relative).any().sum()} приборов;"
        f" хотя бы одно: {(absolute | relative).any().sum()}")
    say("   чувствительность (приборов):",
        {f">{t} м³": int((daily > t).any().sum()) for t in [1, 2, 3, 5]},
        {f">{k}×": int(((daily > k * base) & (daily >= SPIKE_REL_MIN) & (base >= BASE_MIN)).any().sum())
         for k in [5, 10, 20]})
    top_abs = rows[rows.absolute].sort_values("daily_m3", ascending=False).drop_duplicates("anon_id")
    top_rel = rows[rows.relative].sort_values("times_base", ascending=False).drop_duplicates("anon_id")
    say("   крупнейшие по объёму:\n" + table(top_abs.head(5), ["anon_id", "data", "daily_m3"]))
    say("   крупнейшие относительно своей нормы:\n" +
        table(top_rel.head(5), ["anon_id", "data", "daily_m3", "base_m3", "times_base"]))
    rows.sort_values("daily_m3", ascending=False).to_csv(OUT / "2_spikes.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# 3. Нереальные значения
# ---------------------------------------------------------------------------

def p3_unreal(r):
    over = r[r.daily > r.daily_max].copy()
    k = (over.daily / BIT_STEP).round()
    over["k_x_327"] = (k >= 1) & ((over.daily - k * BIT_STEP).abs() <= 1)
    cap = r[r.pokazanie > COUNTER_CAPACITY]

    say("\n3. НЕРЕАЛЬНЫЕ ЗНАЧЕНИЯ  (сырой суточный расход > паспортного максимума за сутки)")
    say(f"   {len(over)} дней у {over.anon_id.nunique()} приборов; типы: {over.groupby('tip').anon_id.nunique().to_dict()}")
    say(f"   из них показание потом вернулось к прежнему уровню (сбой): {over.glitch.sum()} дней")
    say(f"   скачок ровно на k × {BIT_STEP} м³ (±1): {over.k_x_327.sum()} дней у {over[over.k_x_327].anon_id.nunique()} приборов")
    say(f"   без модели в справочнике: {sorted(over[over.dn.isna()].anon_id.unique())}")
    top = over.sort_values("daily", ascending=False).drop_duplicates("anon_id")
    say(table(top.head(6), ["anon_id", "data", "tip", "pokazanie", "daily"]))
    say(f"   показание больше ёмкости счётчика {COUNTER_CAPACITY}: {sorted(cap.anon_id.unique())}")
    over[["anon_id", "data", "tip", "dn", "pokazanie", "daily", "glitch", "k_x_327"]] \
        .sort_values("daily", ascending=False).to_csv(OUT / "3_unreal.csv", index=False)
    return over


# ---------------------------------------------------------------------------
# 4. Утечка
# ---------------------------------------------------------------------------

def p4_leak(r):
    pos = runs(r, r.daily_clean > 0)
    long = pos[pos.days >= RUN_DAYS].copy()
    long["strict"] = (long.min_m3 >= LEAK_FLOOR_SHARE * long.median_m3) & (long.min_m3 >= LEAK_MIN_FLOOR)
    strict = long[long.strict]

    say("\n4. УТЕЧКА  (нет нулевых суток)")
    say(f"   ≥ {RUN_DAYS} суток подряд с расходом > 0: {long.anon_id.nunique()} приборов")
    say("   по длине серии (приборов):", {f"≥{d}": int(pos[pos.days >= d].anon_id.nunique()) for d in [30, 60, 90, 180]})
    say(f"   строгий отбор (минимум серии ≥ {LEAK_FLOOR_SHARE:.0%} её медианы и ≥ {LEAK_MIN_FLOOR} м³): "
        f"{strict.anon_id.nunique()} приборов, ≥90 суток: {strict[strict.days >= 90].anon_id.nunique()}")
    best = strict.sort_values(["days", "min_m3"], ascending=False).drop_duplicates("anon_id")
    say(table(best.head(8), ["anon_id", "start", "end", "days", "min_m3", "median_m3"]))
    long.sort_values(["strict", "days"], ascending=False).to_csv(OUT / "4_leak_runs.csv", index=False)
    return long


# ---------------------------------------------------------------------------
# 5. Нулевое потребление при идущих пакетах
# ---------------------------------------------------------------------------

def p5_zero(r):
    zero = runs(r, (r.daily_clean == 0) & (r.paketov_za_sutki > 0))
    long = zero[zero.days >= RUN_DAYS].copy()
    # Отличаю «ноль с самого начала» от «расходовал и вдруг перестал».
    # «Расходовал» = до начала серии набрал хотя бы 1 м³ (единичные литры не в счёт).
    used_before = []
    clean = r.dropna(subset=["daily_clean"]).groupby("anon_id")
    for row in long.itertuples():
        hist = clean.get_group(row.anon_id)
        used_before.append(hist[hist.data < row.start].daily_clean.sum())
    long["used_before_m3"] = np.round(used_before, 3)
    long["stopped"] = long.used_before_m3 >= 1
    long["packets_per_day"] = (long.packets / long.days).round(1)

    say("\n5. НУЛЕВОЙ РАСХОД ПРИ ИДУЩИХ ПАКЕТАХ")
    say(f"   ≥ {RUN_DAYS} суток подряд расход 0 и пакеты каждый день: {long.anon_id.nunique()} приборов")
    say("   по длине серии (приборов):", {f"≥{d}": int(zero[zero.days >= d].anon_id.nunique()) for d in [30, 60, 90, 180]})
    say(f"   из них раньше расходовали воду и «замерли»: {long[long.stopped].anon_id.nunique()}")
    cols = ["anon_id", "start", "end", "days", "packets", "packets_per_day"]
    say(table(long.sort_values("days", ascending=False).drop_duplicates("anon_id").head(4), cols))
    say("   «замерли» после расхода:\n" +
        table(long[long.stopped].sort_values("days", ascending=False).drop_duplicates("anon_id").head(5), cols))
    long.sort_values("days", ascending=False).to_csv(OUT / "5_zero_runs.csv", index=False)
    return long


# ---------------------------------------------------------------------------
# 6. Ступенька
# ---------------------------------------------------------------------------

def p6_steps(r):
    d = r.dropna(subset=["daily_clean"]).assign(month=lambda x: x.data.dt.to_period("M"))
    m = d.groupby(["anon_id", "month"]).daily_clean.agg(["median", "size"])
    lvl = m[m["size"] >= MONTH_MIN_DAYS]["median"].unstack()    # медиана суток по месяцам
    months = list(lvl.columns)

    found = []
    for k in range(STEP_MONTHS, len(months) - STEP_MONTHS + 1):
        before = lvl[months[k - STEP_MONTHS:k]]
        after = lvl[months[k:k + STEP_MONTHS]]
        full = before.notna().all(axis=1) & after.notna().all(axis=1)
        b_lo, b_hi = before.min(axis=1), before.max(axis=1)
        a_lo, a_hi = after.min(axis=1), after.max(axis=1)
        # «устойчиво» = самый низкий месяц одной стороны выше самого высокого месяца другой
        up = full & (a_lo >= STEP_FACTOR * b_hi.clip(lower=0.01)) & (a_lo - b_hi >= STEP_MIN_DIFF)
        down = full & (b_lo >= STEP_FACTOR * a_hi.clip(lower=0.01)) & (b_lo - a_hi >= STEP_MIN_DIFF)
        for direction, mask in [("рост", up), ("падение", down)]:
            for mid in lvl.index[mask]:
                found.append({"anon_id": mid, "step_month": str(months[k]), "direction": direction,
                              "before_m3": before.loc[mid].median(), "after_m3": after.loc[mid].median()})
    steps = pd.DataFrame(found)
    steps["diff_m3"] = (steps.after_m3 - steps.before_m3).round(3)
    # у прибора может сработать несколько соседних месяцев — оставляю самый сильный
    best = steps.assign(a=steps.diff_m3.abs()).sort_values("a", ascending=False) \
        .drop_duplicates("anon_id").drop(columns="a")
    to_zero = ((best.direction == "падение") & (best.after_m3 < 0.01)).sum()
    from_zero = ((best.direction == "рост") & (best.before_m3 < 0.01)).sum()

    # Для сравнения — вариант прямо из условия: средний расход второй половины vs первой.
    mid_date = d.data.min() + (d.data.max() - d.data.min()) / 2
    halves = d.groupby(["anon_id", d.data >= mid_date]).daily_clean.agg(["mean", "size"]).unstack()
    ok = (halves["size"] >= 60).all(axis=1)
    h1, h2 = halves["mean"][False][ok], halves["mean"][True][ok]
    half_up = ((h2 >= 3 * h1.clip(lower=0.01)) & (h2 - h1 >= STEP_MIN_DIFF)).sum()
    half_down = ((h1 >= 3 * h2.clip(lower=0.01)) & (h1 - h2 >= STEP_MIN_DIFF)).sum()

    say("\n6. СТУПЕНЬКА")
    say(f"   можно оценить {int((lvl.notna().sum(axis=1) >= 2 * STEP_MONTHS).sum())} приборов (≥ 6 полных месяцев)")
    say(f"   найдено {len(best)}: рост {(best.direction == 'рост').sum()}, падение {(best.direction == 'падение').sum()};"
        f" из них «ноль → расход» {from_zero}, «расход → ноль» {to_zero}")
    say(f"   упрощённо (вторая половина года ≥ 3× первой или наоборот): рост {half_up}, падение {half_down} из {ok.sum()}")
    say(table(best.head(8), ["anon_id", "step_month", "direction", "before_m3", "after_m3"]))
    best.to_csv(OUT / "6_steps.csv", index=False)
    return best


# ---------------------------------------------------------------------------
# 7. Магнит
# ---------------------------------------------------------------------------

def p7_magnet(r, events, zero, neg, steps):
    magnet = events[events.sobytie == "магнит"]
    first = magnet.groupby("anon_id").data.min()
    daily = to_grid(r, "daily_clean")

    def before_after(mid, day):
        s = daily[mid]
        pre = s[day - pd.Timedelta(days=MAGNET_WINDOW):day - pd.Timedelta(days=1)].dropna()
        post = s[day:day + pd.Timedelta(days=MAGNET_WINDOW - 1)].dropna()
        if len(pre) < 15 or len(post) < 15 or pre.mean() < 0.05:
            return None
        return pre.mean(), post.mean()

    rows = []
    for mid, day in first.items():
        res = before_after(mid, day)
        if res:
            rows.append({"anon_id": mid, "first_magnet": day, "before_m3": res[0], "after_m3": res[1]})
    mag = pd.DataFrame(rows)
    mag["drop"] = mag.after_m3 <= MAGNET_DROP * mag.before_m3

    # Контроль: приборы без «магнита» тех же типов, та же проверка на случайную дату
    # из тех же дат, что и у первых магнитов. Если падения там столько же — связи нет.
    magnet_types = r[r.anon_id.isin(first.index)].tip.unique()
    pool = r[r.tip.isin(magnet_types) & ~r.anon_id.isin(first.index)].anon_id.unique()
    rng = np.random.default_rng(42)
    dates = rng.choice(first.values, size=len(pool))
    ctrl = [before_after(m, pd.Timestamp(d)) for m, d in zip(pool, dates)]
    ctrl = pd.DataFrame([c for c in ctrl if c], columns=["before_m3", "after_m3"])
    ctrl_drop = (ctrl.after_m3 <= MAGNET_DROP * ctrl.before_m3).mean()

    # Второй взгляд — по приборам целиком: у кого был магнит, чаще ли у них ноль/сбросы/падения?
    meters = r[r.tip.isin(magnet_types)].drop_duplicates("anon_id")[["anon_id"]]
    meters["magnet"] = meters.anon_id.isin(first.index)
    meters["zero_run"] = meters.anon_id.isin(zero[zero.stopped].anon_id)
    meters["negative"] = meters.anon_id.isin(neg[neg.change < -0.01].anon_id)
    meters["step_down"] = meters.anon_id.isin(steps[steps.direction == "падение"].anon_id)
    share = meters.groupby("magnet")[["zero_run", "negative", "step_down"]].mean().mul(100).round(1)

    # Третий взгляд — по дням у самих «магнитных» приборов: в день события расход ниже?
    days = r[r.anon_id.isin(first.index)].dropna(subset=["daily_clean"]) \
        .merge(magnet[["anon_id", "data"]].assign(event=True), on=["anon_id", "data"], how="left")
    days["event"] = days.event.eq(True)
    by_day = days.groupby("event").daily_clean.agg(
        days="size", mean_m3="mean", zero_share=lambda s: (s == 0).mean()).round(3)

    appeared = r.groupby("anon_id").data.min()
    late = first[first >= "2026-09-01"]

    say("\n7. МАГНИТ")
    say(f"   уникальных «прибор-день» с магнитом: {len(magnet)}, приборов: {first.size};"
        f" типы: {r[r.anon_id.isin(first.index)].drop_duplicates('anon_id').tip.value_counts().to_dict()}")
    say(f"   хватает данных за 30 дней до и после первого магнита: {len(mag)} приборов;"
        f" падение в 2+ раза: {mag['drop'].sum()} ({mag['drop'].mean():.1%})")
    say(f"   контроль без магнита (случайная дата): {len(ctrl)} приборов, падение у {ctrl_drop:.1%}")
    say(table(mag[mag["drop"]], ["anon_id", "first_magnet", "before_m3", "after_m3"]))
    say("   доля приборов, % (False — без магнита, True — с магнитом):\n" + share.to_string())
    say("   у «магнитных» приборов: дни с событием (True) и без (False):\n" + by_day.to_string())
    say(f"   первый магнит в сентябре 2026: {len(late)} из {first.size} приборов;"
        f" из них сам прибор появился в данных в сентябре: {(appeared[late.index] >= '2026-09-01').sum()}")
    mag.to_csv(OUT / "7_magnet.csv", index=False)
    return mag, ctrl_drop, share


# ---------------------------------------------------------------------------
# 8. Собственные находки
# ---------------------------------------------------------------------------

def p8_own(r, meters):
    say("\n8. СОБСТВЕННЫЕ НАХОДКИ")

    # 8.1 Сбои AQUA2: скачки ровно на k × 327.68 м³
    jumps = r[(r.gap == 1) & (r.change.abs() > 50)].copy()
    k = (jumps.change.abs() / BIT_STEP).round()
    jumps["k"] = k
    jumps["is_bit"] = (k >= 1) & ((jumps.change.abs() - k * BIT_STEP).abs() <= 1)
    bit = jumps[jumps.is_bit]
    aqua2 = (meters.tip == "AQUA2").sum()
    say(f"   8.1 скачков |Δ| > 50 м³ за сутки: {len(jumps)}; из них ровно k×{BIT_STEP} ± 1 м³: {len(bit)}"
        f" у {bit.anon_id.nunique()} приборов (все AQUA2: {set(bit.tip) == {'AQUA2'}}, всего AQUA2 — {aqua2})")
    say("       k:", bit.k.astype(int).value_counts().sort_index().to_dict())
    say("       чаще всего:", bit.anon_id.value_counts().head(5).to_dict())
    bit.to_csv(OUT / "8_bit_327.csv", index=False)

    # 8.2 Одновременный рост показания ровно в 10 раз
    prev = r.groupby("anon_id").pokazanie.shift()
    x10 = r[(r.gap == 1) & (prev > 1) & (r.pokazanie / prev).between(9.9, 10.1)].copy()
    x10["prev"] = prev[x10.index]
    say(f"   8.2 показание за сутки выросло ровно в 10 раз: {x10.anon_id.nunique()} приборов, "
        f"даты {sorted(x10.data.dt.strftime('%Y-%m-%d').unique())}, типы {sorted(x10.tip.unique())}")
    say(table(x10, ["anon_id", "data", "prev", "pokazanie"]))
    x10.to_csv(OUT / "8_x10.csv", index=False)

    # 8.3 Одно и то же «фантомное» показание у разных приборов.
    # Ищу однодневные выбросы: вчера и завтра показание одинаковое, а сегодня другое.
    g = r.groupby("anon_id").pokazanie
    prev, nxt = g.shift(), g.shift(-1)
    one_day = r[((prev - nxt).abs() <= 0.001) & ((r.pokazanie - prev).abs() > 0.3)].copy()
    one_day["value"] = one_day.pokazanie.round(3)
    shared = one_day.groupby("value").anon_id.nunique()
    shared = shared[shared >= 5]
    phantom = one_day[one_day.value.isin(shared.index)]
    say(f"   8.3 однодневных выбросов «туда и обратно»: {len(one_day)} у {one_day.anon_id.nunique()} приборов;"
        f" одно и то же значение у ≥5 приборов: {shared.to_dict()}")
    say(f"       {sorted(phantom.anon_id.unique())}; типы {phantom.drop_duplicates('anon_id').tip.value_counts().to_dict()},"
        f" БС {phantom.drop_duplicates('anon_id').bs.value_counts().to_dict()}")
    phantom.to_csv(OUT / "8_phantom.csv", index=False)

    # Для справки: откуда приборы с короткой историей
    first = r.groupby("anon_id").agg(first=("data", "min"), bs=("bs", "first"), tip=("tip", "first"))
    late = first[first["first"] >= "2026-09-01"]
    say(f"   справка: {len(late)} приборов впервые появились в сентябре 2026 "
        f"(БС {late.bs.value_counts().head(2).to_dict()}) — это новая партия, а не потеря связи")
    return bit, x10, phantom


# ---------------------------------------------------------------------------
# Кого проверять первым
# ---------------------------------------------------------------------------

def inspection_list(r, neg, spikes, leak, zero, mag, bit, x10, phantom):
    rows = []

    def add(ids, group, reason, action):
        for i in ids:
            rows.append({"anon_id": i, "group": group, "reason": reason, "action": action})

    cap = r[r.pokazanie > COUNTER_CAPACITY].anon_id.unique()
    add(cap, "1. данные", "показание больше ёмкости счётчика (миллионы м³)", "сырые пакеты, декодер")
    add(x10.anon_id.unique(), "1. данные", "показание ×10 за сутки 17.10.2025 у всех шести", "обработка данных за 17.10.2025")
    add(bit.anon_id.value_counts().index[:10], "1. данные", "повторяющиеся скачки на k×327,68 м³ (AQUA2)",
        "прошивка/декодер AQUA2")
    add(phantom.anon_id.unique(), "1. данные", "фантомное показание 0,534 на один день (одно и то же у 7 приборов)",
        "прошивка aqua2_nvt_wa_stm")
    resets = neg[(neg.gap == 1) & neg.kind.str.startswith("уровень")].anon_id.unique()
    add(resets, "1. данные", "показание упало и не вернулось", "замена/сброс прибора, был ли акт")

    strict = leak[leak.strict].sort_values(["days", "min_m3"], ascending=False).drop_duplicates("anon_id")
    add(strict[strict.days >= 90].anon_id, "2. выезд", "≥90 суток без перерыва, «пол» расхода не опускается",
        "проверка на утечку: закрыть краны, смотреть счётчик")
    stuck = zero[zero.stopped].sort_values("days", ascending=False).drop_duplicates("anon_id")
    add(stuck[stuck.days >= 180].anon_id, "2. выезд", "расходовал воду, потом ≥180 суток ровно 0 при живой связи",
        "заклинивший счётчик / обход учёта")
    # 3+ дня ПОДРЯД с расходом в 10+ раз выше нормы — похоже на прорыв, а не на стирку
    rel = spikes[spikes.relative].sort_values(["anon_id", "data"])
    streak = ((rel.anon_id != rel.anon_id.shift()) | (rel.data.diff().dt.days != 1)).cumsum()
    lengths = rel.groupby(streak).agg(anon_id=("anon_id", "first"), days=("data", "size"))
    add(lengths[lengths.days >= 3].anon_id.unique(), "2. выезд",
        "3+ суток подряд расход в 10+ раз выше нормы", "прорыв / разовая утечка")
    add(mag[mag["drop"]].anon_id, "3. проверить", "расход упал в 2+ раза после первого «магнита»",
        "осмотр пломбы и прибора")

    out = pd.DataFrame(rows).drop_duplicates("anon_id")
    out.to_csv(OUT / "inspection_list.csv", index=False)
    say("\nОЧЕРЕДЬ НА ПРОВЕРКУ:", out.groupby("group").size().to_dict())
    return out


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def charts(r, mag, ctrl_drop):
    CHARTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Как выглядит сбой AQUA2
    x = r[(r.anon_id == "8190741") & (r.data >= "2025-12-10") & (r.data <= "2026-02-12")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x.data, x.pokazanie, marker=".", lw=1, color="#444")
    base = x.pokazanie.quantile(0.1)
    for k in range(1, 4):
        ax.axhline(base + k * BIT_STEP, ls=":", color="#d62728", lw=0.8)
        ax.text(x.data.max(), base + k * BIT_STEP, f"  +{k}×327,68", va="center", fontsize=8, color="#d62728")
    ax.set_title("8190741 (AQUA2): показание прыгает ровно на k×327,68 м³ и возвращается", fontsize=10)
    ax.set_ylabel("показание, м³")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m"))
    fig.tight_layout()
    fig.savefig(CHARTS / "aqua2_327.png", dpi=130)
    plt.close(fig)

    # 2. Утечка vs заклинивший счётчик
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, mid, title in [(axes[0], "5754693", "5754693: расход ни дня не опускается ниже 0,75 м³"),
                           (axes[1], "4530662", "4530662: с 22.03 расход ровно 0, пакеты идут (~12 в сутки)")]:
        s = r[r.anon_id == mid].set_index("data").daily_clean
        s = s.reindex(pd.date_range(r.data.min(), r.data.max()))
        ax.plot(s.index, s.values, lw=0.8, color="#1f77b4")
        ax.set_ylim(bottom=-0.02 * s.max())
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("м³/сутки")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "leak_vs_stuck.png", dpi=130)
    plt.close(fig)

    # 3. Магнит vs контроль
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    vals = [mag["drop"].mean() * 100, ctrl_drop * 100]
    ax.bar(["после первого\n«магнита»", "контроль:\nбез магнита"], vals, color=["#d62728", "#999"], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center")
    ax.set_ylabel("% приборов")
    ax.set_title("Расход упал в 2+ раза за 30 дней", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "magnet.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    readings, meters, events = load()
    r = add_daily(readings, meters)
    say(f"Показаний {len(r)}, приборов {r.anon_id.nunique()}, период {r.data.min().date()} – {r.data.max().date()}")
    say(f"Суточных разниц (соседние дни): {r.daily.notna().sum()}; разниц через пропуск (не используются): {(r.gap > 1).sum()}")
    say(f"Медиана суточного расхода: {r.daily_clean.median():.3f} м³, 90-й перцентиль {r.daily_clean.quantile(.9):.3f}, "
        f"99-й {r.daily_clean.quantile(.99):.3f}; суток с нулём: {(r.daily_clean == 0).mean():.0%}")
    say(f"Сбойных показаний: {r.glitch.sum()} у {r[r.glitch].anon_id.nunique()} приборов; "
        f"суток в чистом ряду: {r.daily_clean.notna().sum()}")

    neg = p1_negative(r, events)
    spikes = p2_spikes(r)
    p3_unreal(r)
    leak = p4_leak(r)
    zero = p5_zero(r)
    steps = p6_steps(r)
    mag, ctrl_drop, _ = p7_magnet(r, events, zero, neg, steps)
    bit, x10, phantom = p8_own(r, meters)
    inspection_list(r, neg, spikes, leak, zero, mag, bit, x10, phantom)
    charts(r, mag, ctrl_drop)

    r[["anon_id", "data", "pokazanie", "paketov_za_sutki", "gap", "daily", "glitch", "daily_clean"]] \
        .to_csv(OUT / "daily_consumption.csv.gz", index=False)
    (OUT / "summary.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

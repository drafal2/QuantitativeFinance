# Press the green button in the gutter to run the script.

# https://github.com/sainiabhinav95/quant_models/blob/b9080592097f7fe26f3369e45d7839d92b617c5a/irate_models/hull_white.py
# https://sainiabhinav.medium.com/quant-finance-series-hull-white-model-in-python-b44876723c56

if __name__ == '__main__':
    import QuantLib as ql
    import numpy as np

    # Parametry swapu
    notional = 10e8
    fixed_rate = 0.03
    fixed_freq = ql.Annual
    float_freq = ql.Quarterly
    swap_tenor = 10  # lat

    # Daty
    calendar = ql.TARGET()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    spot = calendar.advance(today, 2, ql.Days)
    maturity = calendar.advance(spot, swap_tenor, ql.Years)

    # Harmonogramy płatności
    fixed_schedule = ql.Schedule(spot, maturity, ql.Period(fixed_freq), calendar,
                                 ql.ModifiedFollowing, ql.ModifiedFollowing,
                                 ql.DateGeneration.Forward, False)
    float_schedule = ql.Schedule(spot, maturity, ql.Period(float_freq), calendar,
                                 ql.ModifiedFollowing, ql.ModifiedFollowing,
                                 ql.DateGeneration.Forward, False)

    # Krzywa zerokuponowa (przykładowa płaska 3%)
    rate = ql.SimpleQuote(0.03)
    flat_curve = ql.FlatForward(spot, ql.QuoteHandle(rate), ql.Actual365Fixed())
    discount_curve = ql.YieldTermStructureHandle(flat_curve)

    # Indeks stopy zmiennej
    euribor = ql.Euribor3M(discount_curve)

    # Tworzenie swapu
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer, notional,
        fixed_schedule, fixed_rate, ql.Actual365Fixed(),
        float_schedule, euribor, 0.0, ql.Actual365Fixed()
    )

    # Engine wyceny
    engine = ql.DiscountingSwapEngine(discount_curve)
    swap.setPricingEngine(engine)

    # Model Hull-White
    hw_model = ql.HullWhite(discount_curve, a=0.03, sigma=0.01)

    # Monte Carlo: parametry
    num_paths = 1000
    time_steps = 40  # co kwartał przez 10 lat

    # Generowanie ścieżek stóp procentowych
    times = np.linspace(0, swap_tenor, time_steps)
    exposures = np.zeros((num_paths, time_steps))

    for i in range(num_paths):
        rates = []
        for t in times:
            r = hw_model.forwardRate(0, t)  # Changed from shortRate to shortForward
            rates.append(r)
        # Aktualizacja krzywej i wycena swapu w każdym punkcie
        for j, t in enumerate(times):
            # Tworzymy nową krzywą z aktualną stopą
            curve = ql.FlatForward(spot, rates[j], ql.Actual365Fixed())
            discount = ql.YieldTermStructureHandle(curve)
            engine = ql.DiscountingSwapEngine(discount)
            swap.setPricingEngine(engine)
            npv = swap.NPV()
            exposures[i, j] = max(npv, 0)  # ekspozycja tylko dodatnia

    # Oblicz PFE (np. percentyl 95%)
    pfe = np.percentile(exposures, 95, axis=0)

    print("PFE w kolejnych terminach (co kwartał):")
    print(pfe)
# Quick validation against Farres & Petersen 2019 --> RUN THIS INSIDE __main__ in main_new.py
# Important: use A = 161 # m^2 and m = 6310 kg to match the paper's A/m = 0.025515 m^2/kg
load_kernels()
et_test = spice.str2et("2026-05-12T00:00:00")
r_sun_test = spice.spkezr("SUN", et_test, "J2000", "NONE", "SSB")[0][:3]
# Place a JWST-like spacecraft at 1.01 AU along +x from the Sun
r_test = r_sun_test + np.array([1.01 * R_SUN_REF_KM, 0.0, 0.0])
m_test = 6310.0   # JWST mass that gives A/m = 161/6310 = 0.025515
a_test = a_SRP(r_test, m_test, et_test)
print(f"|a_SRP| at SEL2 with JWST inputs: {np.linalg.norm(a_test):.4e} km/s^2")
print(f"Paper target (Farres & Petersen 2019): 2.0575e-10 km/s^2")
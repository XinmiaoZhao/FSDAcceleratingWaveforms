from .phenomd_utils import *

from scipy.interpolate import CubicSpline

def chiPN(Seta: float, eta: float, chi1: float, chi2: float) -> float:
    """
    Return the reduced PN spin parameter from Eq. (5.9) of
    arXiv:1107.1267v2.
    """
    chi_s = chi1 + chi2
    chi_a = chi1 - chi2
    return 0.5 * (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)

def next_pow2(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    return 2 ** math.ceil(math.log2(n))

def step_func_boolean(t: float, t1: float) -> bool:
    """Return the Boolean step function ``t >= t1``."""
    return t >= t1

def final_spin0815_s(eta: float, s: float) -> float:
    """
    Predict the final spin using Eq. (3.6) of arXiv:1508.07250.

    The input ``s`` follows the definition in that reference.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    s2 = s * s
    s3 = s2 * s
    return eta * (
        3.4641016151377544 - 4.399247300629289 * eta +
        9.397292189321194 * eta2 - 13.180949901606242 * eta3 +
        s * (
            (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta) +
            (0.1014665242971878 - 2.0967746996832157 * eta) * s +
            (-1.3546806617824356 + 4.108962025369336 * eta) * s2 +
            (-0.8676969352555539 + 2.064046835273906 * eta) * s3
        )
    )

def final_spin0815(eta: float, chi1: float, chi2: float) -> float:
    """
    Compute the final spin for the component-spin convention m1 >= m2.

    ``chi1`` is the spin of the primary mass m1.
    """
    Seta = math.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    s = (m1 * m1 * chi1 + m2 * m2 * chi2)
    return final_spin0815_s(eta, s)

def fring(eta: float, chi1: float, chi2: float, finspin: float) -> float:
    """
    Compute the real ringdown frequency using Fig. 9 of arXiv:1508.07250.

    The interpolation uses the global ``QNMData_a`` and ``QNMData_fring``
    arrays.
    """
    if finspin > 1.0:
        raise ValueError("PhenomD fring function: final spin > 1.0 not supported")

    cs = CubicSpline(QNMData_a, QNMData_fring, bc_type='natural')
    norm_factor = 1.0 - PhenomInternal_EradRational0815(eta, chi1, chi2)
    return cs(finspin) / norm_factor

def fdamp(eta: float, chi1: float, chi2: float, finspin: float) -> float:
    """
    Compute the imaginary ringdown frequency using Fig. 9 of
    arXiv:1508.07250.

    The interpolation uses the global ``QNMData_a`` and ``QNMData_fdamp``
    arrays.
    """
    if finspin > 1.0:
        raise ValueError("PhenomD fdamp function: final spin > 1.0 not supported")

    cs = CubicSpline(QNMData_a, QNMData_fdamp, bc_type='natural')
    norm_factor = 1.0 - PhenomInternal_EradRational0815(eta, chi1, chi2)
    return cs(finspin) / norm_factor

# Amplitude functions

def amp0Func(eta: float) -> float:
    """Return the amplitude scale defined by Eq. (17) of arXiv:1508.07253."""
    return math.sqrt((2.0 / 3.0) * eta) * PI_M_SIXTH


# ///////////////////////////// Amplitude: Inspiral functions /////////////////////////

# // Phenom coefficients rho1, ..., rho3 from direct fit
# // AmpInsDFFitCoeffChiPNFunc[eta, chiPN]

def rho1_fun(eta: float, eta2: float, xi: float) -> float:
    """Return the rho_1 phenomenological coefficient from Table 5."""
    return (3931.8979897196696 - 17395.758706812805 * eta +
            (3132.375545898835 + 343965.86092361377 * eta - 1.2162565819981997e6 * eta2 +
             (-70698.00600428853 + 1.383907177859705e6 * eta - 3.9662761890979446e6 * eta2) * xi +
             (-60017.52423652596 + 803515.1181825735 * eta - 2.091710365941658e6 * eta2) * xi * xi) * xi)

def rho2_fun(eta: float, eta2: float, xi: float) -> float:
    """Return the rho_2 phenomenological coefficient from Table 5."""
    return (-40105.47653771657 + 112253.0169706701 * eta +
            (23561.696065836168 - 3.476180699403351e6 * eta + 1.137593670849482e7 * eta2 +
             (754313.1127166454 - 1.308476044625268e7 * eta + 3.6444584853928134e7 * eta2) * xi +
             (596226.612472288 - 7.4277901143564405e6 * eta + 1.8928977514040343e7 * eta2) * xi * xi) * xi)

def rho3_fun(eta: float, eta2: float, xi: float) -> float:
    """Return the rho_3 phenomenological coefficient from Table 5."""
    return (83208.35471266537 - 191237.7264145924 * eta +
            (-210916.2454782992 + 8.71797508352568e6 * eta - 2.6914942420669552e7 * eta2 +
             (-1.9889806527362722e6 + 3.0888029960154563e7 * eta - 8.390870279256162e7 * eta2) * xi +
             (-1.4535031953446497e6 + 1.7063528990822166e7 * eta - 4.2748659731120914e7 * eta2) * xi * xi) * xi)

def AmpInsAnsatz(Mf: float, powers_of_Mf: UsefulPowers, prefactors: AmpInsPrefactors) -> float:
    """
    Evaluate the re-expanded inspiral amplitude of Eqs. (29)–(30) and
    Appendix B of arXiv:1508.07253.

    ``Mf`` is the dimensionless frequency, ``powers_of_Mf`` stores its
    precomputed powers, and ``prefactors`` stores the inspiral coefficients.
    """
    return (1 +
            powers_of_Mf.two_thirds * prefactors.two_thirds +
            powers_of_Mf.four_thirds * prefactors.four_thirds +
            powers_of_Mf.five_thirds * prefactors.five_thirds +
            powers_of_Mf.seven_thirds * prefactors.seven_thirds +
            powers_of_Mf.eight_thirds * prefactors.eight_thirds +
            Mf * (prefactors.one + Mf * prefactors.two + powers_of_Mf.two * prefactors.three))

def AmpInsAnsatz_vectorized(Mf: np.ndarray, powers_of_Mf: UsefulPowerArrays, prefactors: AmpInsPrefactors) -> np.ndarray:
    """Evaluate ``AmpInsAnsatz`` for a NumPy frequency array."""
    return (1 +
            powers_of_Mf.two_thirds * prefactors.two_thirds +
            powers_of_Mf.four_thirds * prefactors.four_thirds +
            powers_of_Mf.five_thirds * prefactors.five_thirds +
            powers_of_Mf.seven_thirds * prefactors.seven_thirds +
            powers_of_Mf.eight_thirds * prefactors.eight_thirds +
            Mf * (prefactors.one + Mf * prefactors.two + powers_of_Mf.two * prefactors.three))

def init_amp_ins_prefactors(prefactors: AmpInsPrefactors, p: IMRPhenomDAmplitudeCoefficients) -> int:
    """Populate inspiral-amplitude prefactors from coefficient structure ``p``."""
    if p is None:
        raise ValueError("p is None")
    if prefactors is None:
        raise ValueError("prefactors is None")

    eta = p.eta
    prefactors.amp0 = amp0Func(eta)

    chi1 = p.chi1
    chi2 = p.chi2
    chi12 = p.chi12
    chi22 = p.chi22
    eta2 = p.eta2
    eta3 = p.eta3
    Pi = LAL_PI
    Pi2 = powers_of_pi["two"]
    Seta = p.Seta
    SetaPlus1 = p.SetaPlus1

    prefactors.two_thirds = ((-969 + 1804 * eta) * powers_of_pi["two_thirds"]) / 672.0
    prefactors.one = ((chi1 * (81 * SetaPlus1 - 44 * eta) + chi2 * (81 - 81 * Seta - 44 * eta)) * Pi) / 48.0
    prefactors.four_thirds = ((-27312085.0 - 10287648 * chi22 - 10287648 * chi12 * SetaPlus1 + 10287648 * chi22 * Seta +
                                24 * (-1975055 + 857304 * chi12 - 994896 * chi1 * chi2 + 857304 * chi22) * eta +
                                35371056 * eta2) * powers_of_pi["four_thirds"]) / 8.128512e6
    prefactors.five_thirds = (powers_of_pi["five_thirds"] *
                              (chi2 * (-285197 * (-1 + Seta) + 4 * (-91902 + 1579 * Seta) * eta - 35632 * eta2) +
                               chi1 * (285197 * SetaPlus1 - 4 * (91902 + 1579 * Seta) * eta - 35632 * eta2) +
                               42840 * (-1.0 + 4 * eta) * Pi)
                              ) / 32256.0
    prefactors.two = - (Pi2 * (
        -336 * (-3248849057.0 + 2943675504 * chi12 - 3339284256 * chi1 * chi2 + 2943675504 * chi22) * eta2 -
        324322727232 * eta3 -
        7 * (-177520268561 + 107414046432 * chi22 + 107414046432 * chi12 * SetaPlus1 - 107414046432 * chi22 * Seta +
             11087290368 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * Pi) +
        12 * eta * (-545384828789 - 176491177632 * chi1 * chi2 + 202603761360 * chi22 +
                    77616 * chi12 * (2610335 + 995766 * Seta) - 77287373856 * chi22 * Seta +
                    5841690624 * (chi1 + chi2) * Pi + 21384760320 * Pi2)
    )) / 6.0085960704e10
    prefactors.seven_thirds = p.rho1
    prefactors.eight_thirds = p.rho2
    prefactors.three = p.rho3

    return 0

def DAmpInsAnsatz(Mf: float, powers_of_Mf: UsefulPowers, p: IMRPhenomDAmplitudeCoefficients) -> float:
    """
    Evaluate the frequency derivative ``d AmpInsAnsatz / d Mf``.

    ``powers_of_Mf`` contains precomputed powers of the dimensionless
    frequency and ``p`` contains the amplitude coefficients.
    """
    eta = p.eta
    chi1 = p.chi1
    chi2 = p.chi2
    chi12 = p.chi12
    chi22 = p.chi22
    eta2 = p.eta2
    eta3 = p.eta3
    Pi = LAL_PI
    Pi2 = powers_of_pi["two"]
    Seta = p.Seta
    SetaPlus1 = p.SetaPlus1

    term1 = ((-969 + 1804 * eta) * powers_of_pi["two_thirds"]) / (1008.0 * powers_of_Mf.third)
    term2 = ((chi1 * (81 * SetaPlus1 - 44 * eta) + chi2 * (81 - 81 * Seta - 44 * eta)) * Pi) / 48.0
    term3 = ((-27312085 - 10287648 * chi22 - 10287648 * chi12 * SetaPlus1 +
              10287648 * chi22 * Seta + 24 * (-1975055 + 857304 * chi12 - 994896 * chi1 * chi2 + 857304 * chi22) * eta +
              35371056 * eta2) * powers_of_Mf.third * powers_of_pi["four_thirds"]) / 6.096384e6
    term4 = (5 * powers_of_Mf.two_thirds * powers_of_pi["five_thirds"] *
             (chi2 * (-285197 * (-1 + Seta) + 4 * (-91902 + 1579 * Seta) * eta - 35632 * eta2) +
              chi1 * (285197 * SetaPlus1 - 4 * (91902 + 1579 * Seta) * eta - 35632 * eta2) +
              42840 * (-1 + 4 * eta) * Pi)) / 96768.0
    term5 = - (Mf * Pi2 * (
            -336 * (-3248849057.0 + 2943675504 * chi12 - 3339284256 * chi1 * chi2 + 2943675504 * chi22) * eta2 -
            324322727232 * eta3 -
            7 * (-177520268561 + 107414046432 * chi22 + 107414046432 * chi12 * SetaPlus1 - 107414046432 * chi22 * Seta +
                 11087290368 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * Pi) +
            12 * eta * (-545384828789.0 - 176491177632 * chi1 * chi2 + 202603761360 * chi22 +
                        77616 * chi12 * (2610335 + 995766 * Seta) - 77287373856 * chi22 * Seta +
                        5841690624 * (chi1 + chi2) * Pi + 21384760320 * Pi2)
    )) / 3.0042980352e10
    term6 = (7.0 / 3.0) * powers_of_Mf.four_thirds * p.rho1
    term7 = (8.0 / 3.0) * powers_of_Mf.five_thirds * p.rho2
    term8 = 3.0 * powers_of_Mf.two * p.rho3

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

# /////////////////////////// Amplitude: Merger-Ringdown functions ///////////////////////

# // Phenom coefficients gamma1, ..., gamma3
# // AmpMRDAnsatzFunc[]

def gamma1_fun(eta: float, eta2: float, xi: float) -> float:
    """
    gamma1 phenom coefficient. See Table 5 in arXiv:1508.07253.
    """
    return (0.006927402739328343 + 0.03020474290328911 * eta +
            (0.006308024337706171 - 0.12074130661131138 * eta + 0.26271598905781324 * eta2 +
             (0.0034151773647198794 - 0.10779338611188374 * eta + 0.27098966966891747 * eta2) * xi +
             (0.0007374185938559283 - 0.02749621038376281 * eta + 0.0733150789135702 * eta2) * xi * xi) * xi)

def gamma2_fun(eta: float, eta2: float, xi: float) -> float:
    """
    gamma2 phenom coefficient. See Table 5 in arXiv:1508.07253.
    """
    return (1.010344404799477 + 0.0008993122007234548 * eta +
            (0.283949116804459 - 4.049752962958005 * eta + 13.207828172665366 * eta2 +
             (0.10396278486805426 - 7.025059158961947 * eta + 24.784892370130475 * eta2) * xi +
             (0.03093202475605892 - 2.6924023896851663 * eta + 9.609374464684983 * eta2) * xi * xi) * xi)

def gamma3_fun(eta: float, eta2: float, xi: float) -> float:
    """
    gamma3 phenom coefficient. See Table 5 in arXiv:1508.07253.
    """
    return (1.3081615607036106 - 0.005537729694807678 * eta +
            (-0.06782917938621007 - 0.6689834970767117 * eta + 3.403147966134083 * eta2 +
             (-0.05296577374411866 - 0.9923793203111362 * eta + 4.820681208409587 * eta2) * xi +
             (-0.006134139870393713 - 0.38429253308696365 * eta + 1.7561754421985984 * eta2) * xi * xi) * xi)

def AmpMRDAnsatz(f: float, p: IMRPhenomDAmplitudeCoefficients) -> float:
    """
    Ansatz for the merger‐ringdown amplitude. Equation 19 in arXiv:1508.07253.

    ``p`` contains the IMRPhenomD amplitude coefficients.
    """
    fRD = p.fRD
    fDM = p.fDM
    gamma1 = p.gamma1
    gamma2 = p.gamma2
    gamma3 = p.gamma3
    fDMgamma3 = fDM * gamma3
    fminfRD = f - fRD
    return math.exp(- (fminfRD) * gamma2 / fDMgamma3) * (fDMgamma3 * gamma1) / ((fminfRD)**2 + (fDMgamma3)**2)

def AmpMRDAnsatz_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """Evaluate ``AmpMRDAnsatz`` for a NumPy frequency array."""
    fRD = p.fRD
    fDM = p.fDM
    gamma1 = p.gamma1
    gamma2 = p.gamma2
    gamma3 = p.gamma3
    fDMgamma3 = fDM * gamma3
    fminfRD = f - fRD

    return np.exp(- (fminfRD) * gamma2 / fDMgamma3) * (fDMgamma3 * gamma1) / ((fminfRD)**2 + (fDMgamma3)**2)

def DAmpMRDAnsatz(f: float, p: IMRPhenomDAmplitudeCoefficients) -> float:
    """
    first frequency derivative of AmpMRDAnsatz.
    """
    fRD = p.fRD
    fDM = p.fDM
    gamma1 = p.gamma1
    gamma2 = p.gamma2
    gamma3 = p.gamma3

    fDMgamma3 = fDM * gamma3
    pow2_fDMgamma3 = (fDMgamma3)**2
    fminfRD = f - fRD
    expfactor = math.exp((fminfRD * gamma2) / fDMgamma3)
    pow2plus = (fminfRD)**2 + pow2_fDMgamma3

    return ((-2 * fDM * fminfRD * gamma3 * gamma1) / pow2plus - (gamma2 * gamma1)) / (expfactor * pow2plus)

def fmaxCalc(p: IMRPhenomDAmplitudeCoefficients) -> float:
    """
    Equation 20 in arXiv:1508.07253 (f_peak): analytic location of maximum of AmpMRDAnsatz.
    If gamma2 >= 1, the square-root term is set to zero.
    """
    fRD = p.fRD
    fDM = p.fDM
    gamma2 = p.gamma2
    gamma3 = p.gamma3

    if gamma2 <= 1:
        return abs(fRD + (fDM * (-1 + math.sqrt(1 - (gamma2)**2)) * gamma3) / gamma2)
    else:
        return abs(fRD + (-fDM * gamma3) / gamma2)


# ///////////////////////////// Amplitude: Intermediate functions ////////////////////////

# // Phenom coefficients delta0, ..., delta4 determined from collocation method
# // (constraining 3 values and 2 derivatives)
# // AmpIntAnsatzFunc[]

def AmpIntAnsatz(Mf: float, p: IMRPhenomDAmplitudeCoefficients) -> float:
    """
    Ansatz for the intermediate amplitude. Equation 21 in arXiv:1508.07253.

    ``p`` supplies coefficients ``delta0`` through ``delta4``.
    """
    Mf2 = Mf * Mf
    return p.delta0 + Mf * p.delta1 + Mf2 * (p.delta2 + Mf * p.delta3 + p.delta4 * Mf2)

def AmpIntAnsatz_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """Evaluate ``AmpIntAnsatz`` for a NumPy frequency array."""
    Mf2 = Mf * Mf
    return p.delta0 + Mf * p.delta1 + Mf2 * (p.delta2 + Mf * p.delta3 + p.delta4 * Mf2)

def AmpIntColFitCoeff(eta: float, eta2: float, chi: float) -> float:
    """
    Amplitude Intermediate Collocation Fit Coefficient (v2 value in Table 5, arXiv:1508.07253).
    xi = -1 + chi.
    """
    xi = -1.0 + chi
    return (0.8149838730507785 + 2.5747553517454658 * eta +
            (1.1610198035496786 - 2.3627771785551537 * eta + 6.771038707057573 * eta2 +
             (0.7570782938606834 - 2.7256896890432474 * eta + 7.1140380397149965 * eta2) * xi +
             (0.1766934149293479 - 0.7978690983168183 * eta + 2.1162391502005153 * eta2) * xi * xi) * xi)

def delta0_fun(p: IMRPhenomDAmplitudeCoefficients, d: DeltaUtility) -> float:
    f1 = p.f1
    f2 = p.f2
    f3 = p.f3
    v1 = p.v1
    v2 = p.v2
    v3 = p.v3
    d1 = p.d1
    d2 = p.d2

    f12 = d.f12
    f13 = d.f13
    f14 = d.f14
    f15 = d.f15
    f22 = d.f22
    f23 = d.f23
    f24 = d.f24
    f32 = d.f32
    f33 = d.f33
    f34 = d.f34
    f35 = d.f35

    num = (d2*f15*f22*f3 - 2*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32
        - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33
        + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33
        - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35
        + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1
        + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2
        + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3
        + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3
        + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3)
    denom = ( (f1 - f2)**2 * (f1 - f3)**3 * (f3 - f2)**2 )
    return - (num / denom)

def delta1_fun(p: IMRPhenomDAmplitudeCoefficients, d: DeltaUtility) -> float:
    f1 = p.f1
    f2 = p.f2
    f3 = p.f3
    v1 = p.v1
    v2 = p.v2
    v3 = p.v3
    d1 = p.d1
    d2 = p.d2

    f12 = d.f12
    f13 = d.f13
    f14 = d.f14
    f15 = d.f15
    f22 = d.f22
    f23 = d.f23
    f24 = d.f24
    f32 = d.f32
    f33 = d.f33
    f34 = d.f34
    f35 = d.f35

    num = (-(d2*f15*f22) + 2*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2*d1*f13*f23*f3
        + 2*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32
        - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32
        - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33
        + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35
        - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1
        + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3
        + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3)
    denom = ( (f1 - f2)**2 * (f1 - f3)**3 * (f2 - f3)**2 )
    return - (num / denom)

def delta2_fun(p: IMRPhenomDAmplitudeCoefficients, d: DeltaUtility) -> float:
    f1 = p.f1
    f2 = p.f2
    f3 = p.f3
    v1 = p.v1
    v2 = p.v2
    v3 = p.v3
    d1 = p.d1
    d2 = p.d2

    f12 = d.f12
    f13 = d.f13
    f14 = d.f14
    f15 = d.f15
    f23 = d.f23
    f24 = d.f24
    f32 = d.f32
    f33 = d.f33
    f34 = d.f34
    f35 = d.f35

    num = (d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2*d2*f12*f24 - d2*f15*f3
        + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32
        + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32
        - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33
        + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1
        - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2
        - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3
        + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3)
    denom = ( (f1 - f2)**2 * (f1 - f3)**3 * (f2 - f3)**2 )
    return - (num / denom)

def delta3_fun(p: IMRPhenomDAmplitudeCoefficients, d: DeltaUtility) -> float:
    f1 = p.f1
    f2 = p.f2
    f3 = p.f3
    v1 = p.v1
    v2 = p.v2
    v3 = p.v3
    d1 = p.d1
    d2 = p.d2

    f12 = d.f12
    f13 = d.f13
    f14 = d.f14
    f22 = d.f22
    f24 = d.f24
    f32 = d.f32
    f33 = d.f33
    f34 = d.f34

    num = (-2*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3
        - 2*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3
        + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32
        + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33
        - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1
        - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2
        - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3
        - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3)
    denom = ( (f1 - f2)**2 * (f1 - f3)**3 * (f2 - f3)**2 )
    return - (num / denom)

def delta4_fun(p: IMRPhenomDAmplitudeCoefficients, d: DeltaUtility) -> float:
    f1 = p.f1
    f2 = p.f2
    f3 = p.f3
    v1 = p.v1
    v2 = p.v2
    v3 = p.v3
    d1 = p.d1
    d2 = p.d2

    f12 = d.f12
    f13 = d.f13
    f22 = d.f22
    f23 = d.f23
    f32 = d.f32
    f33 = d.f33

    num = (d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2*d1*f12*f2*f3
        + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32
        - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1
        - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2
        + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3)
    denom = ( (f1 - f2)**2 * (f1 - f3)**3 * (f2 - f3)**2 )
    return - (num / denom)

def ComputeDeltasFromCollocation(p: IMRPhenomDAmplitudeCoefficients) -> None:
    """
    Compute intermediate-amplitude coefficients ``delta0`` through ``delta4``.

    ``p.fmaxCalc`` is the merger-ringdown amplitude peak, while ``eta``,
    ``eta2``, and the effective-spin parameter ``chi`` determine the fitted
    collocation value. The routine stores the collocation frequencies, values,
    derivatives, and resulting delta coefficients in ``p``.
    """

    f1 = AMP_fJoin_INS
    f3 = p.fmaxCalc
    dfx = 0.5 * (f3 - f1)
    f2 = f1 + dfx

    # Evaluate the inspiral value and derivative at the first join.
    powers_of_f1 = init_useful_powers(f1)
    prefactors = AmpInsPrefactors()
    init_amp_ins_prefactors(prefactors, p)

    v1 = AmpInsAnsatz(f1, powers_of_f1, prefactors)
    d1 = DAmpInsAnsatz(f1, powers_of_f1, p)
    # Evaluate the merger-ringdown value and derivative at the second join.
    v3 = AmpMRDAnsatz(f3, p)
    d2 = DAmpMRDAnsatz(f3, p)
    # The intermediate collocation value v2 is fitted in Table 5.
    v2 = AmpIntColFitCoeff(p.eta, p.eta2, p.chi)

    # Store the collocation inputs.
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.v1 = v1
    p.v2 = v2
    p.v3 = v3
    p.d1 = d1
    p.d2 = d2

    # Precompute powers used by the closed-form delta expressions.
    d = DeltaUtility()
    d.f12 = f1 * f1
    d.f13 = f1 * d.f12
    d.f14 = f1 * d.f13
    d.f15 = f1 * d.f14
    d.f22 = f2 * f2
    d.f23 = f2 * d.f22
    d.f24 = f2 * d.f23
    d.f32 = f3 * f3
    d.f33 = f3 * d.f32
    d.f34 = f3 * d.f33
    d.f35 = f3 * d.f34

    # Compute and store the intermediate polynomial coefficients.
    p.delta0 = delta0_fun(p, d)
    p.delta1 = delta1_fun(p, d)
    p.delta2 = delta2_fun(p, d)
    p.delta3 = delta3_fun(p, d)
    p.delta4 = delta4_fun(p, d)


# ///////////////////////////// Amplitude: glueing function ////////////////////////////

def ComputeIMRPhenomDAmplitudeCoefficients(p: IMRPhenomDAmplitudeCoefficients, eta: float, chi1: float, chi2: float, finspin: float) -> None:
    """
    Populate all IMRPhenomD amplitude coefficients in ``p``.

    This corresponds to ``ComputeIMRPhenomDAmplitudeCoefficients`` in the
    reference C implementation.
    """
    p.eta = eta
    p.etaInv = 1.0 / eta
    p.chi1 = chi1
    p.chi2 = chi2
    p.chi12 = chi1 * chi1
    p.chi22 = chi2 * chi2
    eta2 = eta * eta
    p.eta2 = eta2
    p.eta3 = eta * eta2
    Seta = math.sqrt(1.0 - 4.0 * eta)
    p.Seta = Seta
    p.SetaPlus1 = 1.0 + Seta

    p.q = 0.5 * (1.0 + Seta - 2.0 * eta) * p.etaInv
    p.chi = chiPN(Seta, eta, chi1, chi2)
    xi = -1.0 + p.chi

    p.fRD = fring(eta, chi1, chi2, finspin)
    p.fDM = fdamp(eta, chi1, chi2, finspin)

    # Merger-ringdown coefficients.
    p.gamma1 = gamma1_fun(eta, eta2, xi)
    p.gamma2 = gamma2_fun(eta, eta2, xi)
    p.gamma3 = gamma3_fun(eta, eta2, xi)

    p.fmaxCalc = fmaxCalc(p)

    # Inspiral coefficients.
    p.rho1 = rho1_fun(eta, eta2, xi)
    p.rho2 = rho2_fun(eta, eta2, xi)
    p.rho3 = rho3_fun(eta, eta2, xi)

    # Intermediate coefficients from collocation.
    ComputeDeltasFromCollocation(p)

def IMRPhenDAmplitude(f: float, p: IMRPhenomDAmplitudeCoefficients, powers_of_f: UsefulPowers, prefactors: AmpInsPrefactors) -> float:
    """
    Evaluate the complete IMR amplitude in the region containing ``f``.

    ``p`` holds the amplitude coefficients, ``powers_of_f`` holds precomputed
    frequency powers, and ``prefactors`` holds the inspiral prefactors.
    """
    # Set the amplitude transition frequencies.
    p.fInsJoin = AMP_fJoin_INS
    p.fMRDJoin = p.fmaxCalc

    AmpPreFac = prefactors.amp0 * powers_of_f.m_seven_sixths

    if not step_func_boolean(f, p.fInsJoin):
        AmpIns = AmpInsAnsatz(f, powers_of_f, prefactors)
        return AmpPreFac * AmpIns

    if step_func_boolean(f, p.fMRDJoin):
        AmpMRD = AmpMRDAnsatz(f, p)
        return AmpPreFac * AmpMRD

    AmpInt = AmpIntAnsatz(f, p)
    return AmpPreFac * AmpInt

def IMRPhenDAmplitude_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients, powers_of_f: UsefulPowerArrays, prefactors: AmpInsPrefactors) -> np.ndarray:
    """Evaluate ``IMRPhenDAmplitude`` for a NumPy frequency array."""
    # Set the amplitude transition frequencies.
    p.fInsJoin = AMP_fJoin_INS
    p.fMRDJoin = p.fmaxCalc
    if p.fMRDJoin < p.fInsJoin:
        raise ValueError("fMRDJoin must be greater than fInsJoin")

    AmpPreFac = prefactors.amp0 * powers_of_f.m_seven_sixths

    freqins, freqint, freqmr = split_freqs_amp(f, p)
    AmpIns, AmpInt, AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
    if len(freqins) > 0:
        powers_of_freqins = init_useful_power_arrays(freqins)
        AmpIns = AmpInsAnsatz_vectorized(freqins, powers_of_freqins, prefactors)
    if len(freqint) > 0:
        AmpInt = AmpIntAnsatz_vectorized(freqint, p)
    if len(freqmr) > 0:
        AmpMRD = AmpMRDAnsatz_vectorized(freqmr, p)
    # Concatenate the three frequency regions and apply the common prefactor.
    Amp = np.concatenate((AmpIns, AmpInt, AmpMRD))
    return AmpPreFac * Amp



# /********************************* Phase functions *********************************/

# ////////////////////////////// Phase: Ringdown functions ///////////////////////////

# // alpha_i i=1,2,3,4,5 are the phenomenological intermediate coefficients depending on eta and chiPN
# // PhiRingdownAnsatz is the ringdown phasing in terms of the alpha_i coefficients


def alpha1Fit(eta: float, eta2: float, xi: float) -> float:
    return (43.31514709695348 + 638.6332679188081 * eta +
            ((-32.85768747216059 + 2415.8938269370315 * eta - 5766.875169379177 * eta2) +
             (-61.85459307173841 + 2953.967762459948 * eta - 8986.29057591497 * eta2) * xi +
             (-21.571435779762044 + 981.2158224673428 * eta - 3239.5664895930286 * eta2) * xi * xi) * xi)

def alpha2Fit(eta: float, eta2: float, xi: float) -> float:
    return (-0.07020209449091723 - 0.16269798450687084 * eta +
            ((-0.1872514685185499 + 1.138313650449945 * eta - 2.8334196304430046 * eta2) +
             (-0.17137955686840617 + 1.7197549338119527 * eta - 4.539717148261272 * eta2) * xi +
             (-0.049983437357548705 + 0.6062072055948309 * eta - 1.682769616644546 * eta2) * xi * xi) * xi)

def alpha3Fit(eta: float, eta2: float, xi: float) -> float:
    return (9.5988072383479 - 397.05438595557433 * eta +
            ((16.202126189517813 - 1574.8286986717037 * eta + 3600.3410843831093 * eta2) +
             (27.092429659075467 - 1786.482357315139 * eta + 5152.919378666511 * eta2) * xi +
             (11.175710130033895 - 577.7999423177481 * eta + 1808.730762932043 * eta2) * xi * xi) * xi)

def alpha4Fit(eta: float, eta2: float, xi: float) -> float:
    return (-0.02989487384493607 + 1.4022106448583738 * eta +
            ((-0.07356049468633846 + 0.8337006542278661 * eta + 0.2240008282397391 * eta2) +
             (-0.055202870001177226 + 0.5667186343606578 * eta + 0.7186931973380503 * eta2) * xi +
             (-0.015507437354325743 + 0.15750322779277187 * eta + 0.21076815715176228 * eta2) * xi * xi) * xi)

def alpha5Fit(eta: float, eta2: float, xi: float) -> float:
    return (0.9974408278363099 - 0.007884449714907203 * eta +
            ((-0.059046901195591035 + 1.3958712396764088 * eta - 4.516631601676276 * eta2) +
             (-0.05585343136869692 + 1.7516580039343603 * eta - 5.990208965347804 * eta2) * xi +
             (-0.017945336522161195 + 0.5965097794825992 * eta - 2.0608879367971804 * eta2) * xi * xi) * xi)

def PhiMRDAnsatzInt(f: float, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> float:
    """
    Ringdown phasing ansatz, Equation 14 in arXiv:1508.07253.

    ``p`` supplies ``alpha1`` through ``alpha5`` together with the ringdown
    and damping frequencies.
    """
    sqrootf = math.sqrt(f)
    fpow1_5 = f * sqrootf
    fpow0_75 = math.sqrt(fpow1_5)  # f**0.75
    return ( - (p.alpha2 / f)
             + (4.0 / 3.0) * (p.alpha3 * fpow0_75)
             + p.alpha1 * f
             + p.alpha4 * Rholm * math.atan((f - p.alpha5 * p.fRD) / (Rholm * p.fDM * Taulm)) )

def PhiMRDAnsatzInt_vectorized(f: np.ndarray, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """Evaluate ``PhiMRDAnsatzInt`` for a NumPy frequency array."""
    sqrootf = np.sqrt(f)
    fpow1_5 = f * sqrootf
    fpow0_75 = np.sqrt(fpow1_5)  # f**0.75
    return ( - (p.alpha2 / f)
             + (4.0 / 3.0) * (p.alpha3 * fpow0_75)
             + p.alpha1 * f
             + p.alpha4 * Rholm * np.arctan((f - p.alpha5 * p.fRD) / (Rholm * p.fDM * Taulm)) )

def DPhiMRD(f: float, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> float:
    """Evaluate the first frequency derivative of the ringdown phase ansatz."""
    term1 = p.alpha1
    term2 = p.alpha2 / (f**2)
    term3 = p.alpha3 / (f**0.25)
    denom = p.fDM * Taulm * (1 + ((f - p.alpha5 * p.fRD)**2) / ((p.fDM * Taulm * Rholm)**2))
    term4 = p.alpha4 / denom
    return (term1 + term2 + term3 + term4) * p.etaInv

# ///////////////////////////// Phase: Intermediate functions /////////////////////////////

# // beta_i i=1,2,3 are the phenomenological intermediate coefficients depending on eta and chiPN
# // PhiIntAnsatz is the intermediate phasing in terms of the beta_i coefficients


# // \[Beta]1Fit = PhiIntFitCoeff\[Chi]PNFunc[\[Eta], \[Chi]PN][[1]]

def beta1Fit(eta: float, eta2: float, xi: float) -> float:
    return (97.89747327985583 - 42.659730877489224 * eta +
            ((153.48421037904913 - 1417.0620760768954 * eta + 2752.8614143665027 * eta2) +
             (138.7406469558649 - 1433.6585075135881 * eta + 2857.7418952430758 * eta2) * xi +
             (41.025109467376126 - 423.680737974639 * eta + 850.3594335657173 * eta2) * xi * xi) * xi)

def beta2Fit(eta: float, eta2: float, xi: float) -> float:
    return (-3.282701958759534 - 9.051384468245866 * eta +
            ((-12.415449742258042 + 55.4716447709787 * eta - 106.05109938966335 * eta2) +
             (-11.953044553690658 + 76.80704618365418 * eta - 155.33172948098394 * eta2) * xi +
             (-3.4129261592393263 + 25.572377569952536 * eta - 54.408036707740465 * eta2) * xi * xi) * xi)

def beta3Fit(eta: float, eta2: float, xi: float) -> float:
    return (-0.000025156429818799565 + 0.000019750256942201327 * eta +
            ((-0.000018370671469295915 + 0.000021886317041311973 * eta + 0.00008250240316860033 * eta2) +
             (7.157371250566708e-6 - 0.000055780000112270685 * eta + 0.00019142082884072178 * eta2) * xi +
             (5.447166261464217e-6 - 0.00003220610095021982 * eta + 0.00007974016714984341 * eta2) * xi * xi) * xi)

def PhiIntAnsatz(Mf: float, p: IMRPhenomDPhaseCoefficients) -> float:
    """
    Intermediate phase ansatz, Equation 16 in arXiv:1508.07253.

    ``p`` supplies ``beta1`` through ``beta3``. The factor ``1 / eta`` is
    applied separately when the phase regions are joined.
    """
    return p.beta1 * Mf - p.beta3 / (3.0 * (Mf**3)) + p.beta2 * math.log(Mf)

def PhiIntAnsatz_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """Evaluate ``PhiIntAnsatz`` for a NumPy frequency array."""
    return p.beta1 * Mf - p.beta3 / (3.0 * (Mf**3)) + p.beta2 * np.log(Mf)

def DPhiIntAnsatz(Mf: float, p: IMRPhenomDPhaseCoefficients) -> float:
    """Differentiate the intermediate phase, including the ``1 / eta`` factor."""
    return (p.beta1 + p.beta3 / (Mf**4) + p.beta2 / Mf) * p.etaInv

def DPhiIntTemp(ff: float, p: IMRPhenomDPhaseCoefficients) -> float:
    """Evaluate the shifted intermediate derivative used for phase continuity."""
    return p.C2Int + (p.beta1 + p.beta3 / (ff**4) + p.beta2 / ff) * p.etaInv


# ///////////////////////////// Phase: Inspiral functions /////////////////////////////

# // sigma_i i=1,2,3,4 are the phenomenological inspiral coefficients depending on eta and chiPN
# // PhiInsAnsatzInt is a souped up TF2 phasing which depends on the sigma_i coefficients

def sigma1Fit(eta: float, eta2: float, xi: float) -> float:
    return (2096.551999295543 + 1463.7493168261553 * eta +
            (1312.5493286098522 + 18307.330017082117 * eta - 43534.1440746107 * eta2 +
             (-833.2889543511114 + 32047.31997183187 * eta - 108609.45037520859 * eta2) * xi +
             (452.25136398112204 + 8353.439546391714 * eta - 44531.3250037322 * eta2) * xi * xi) * xi)

def sigma2Fit(eta: float, eta2: float, xi: float) -> float:
    return (-10114.056472621156 - 44631.01109458185 * eta +
            (-6541.308761668722 - 266959.23419307504 * eta + 686328.3229317984 * eta2 +
             (3405.6372187679685 - 437507.7208209015 * eta + 1.6318171307344697e6 * eta2) * xi +
             (-7462.648563007646 - 114585.25177153319 * eta + 674402.4689098676 * eta2) * xi * xi) * xi)

def sigma3Fit(eta: float, eta2: float, xi: float) -> float:
    return (22933.658273436497 + 230960.00814979506 * eta +
            (14961.083974183695 + 1.1940181342318142e6 * eta - 3.1042239693052764e6 * eta2 +
             (-3038.166617199259 + 1.8720322849093592e6 * eta - 7.309145012085539e6 * eta2) * xi +
             (42738.22871475411 + 467502.018616601 * eta - 3.064853498512499e6 * eta2) * xi * xi) * xi)

def sigma4Fit(eta: float, eta2: float, xi: float) -> float:
    return (-14621.71522218357 - 377812.8579387104 * eta +
            (-9608.682631509726 - 1.7108925257214056e6 * eta + 4.332924601416521e6 * eta2 +
             (-22366.683262266528 - 2.5019716386377467e6 * eta + 1.0274495902259542e7 * eta2) * xi +
             (-85360.30079034246 - 570025.3441737515 * eta + 4.396844346849777e6 * eta2) * xi * xi) * xi)

def PhiInsAnsatzInt(Mf: float, powers_of_Mf: UsefulPowers, prefactors: PhiInsPrefactors, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries) -> float:
    if pn is None:
        raise ValueError("pn is None")
    # Assemble PN phasing series
    # v = powers_of_Mf.third * powers_of_pi["third"]
    # logv = math.log(v)
    phasing = prefactors.initial_phasing
    phasing += prefactors.two_thirds * powers_of_Mf.two_thirds
    phasing += prefactors.third * powers_of_Mf.third
    phasing += prefactors.third_with_logv * powers_of_Mf.logv * powers_of_Mf.third
    phasing += prefactors.logv * powers_of_Mf.logv
    phasing += prefactors.minus_third * powers_of_Mf.m_third
    phasing += prefactors.minus_two_thirds * powers_of_Mf.m_two_thirds
    phasing += prefactors.minus_one * powers_of_Mf.inv
    phasing += prefactors.minus_four_thirds / powers_of_Mf.four_thirds
    phasing += prefactors.minus_five_thirds * powers_of_Mf.m_five_thirds
    phasing += (prefactors.one * Mf + prefactors.four_thirds * powers_of_Mf.four_thirds +
                prefactors.five_thirds * powers_of_Mf.five_thirds +
                prefactors.two * powers_of_Mf.two) * p.etaInv
    return phasing

def PhiInsAnsatzInt_vectorized(Mf: np.ndarray, powers_of_Mf: UsefulPowerArrays, prefactors: PhiInsPrefactors, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries) -> np.ndarray:
    if pn is None:
        raise ValueError("pn is None")
    # Assemble PN phasing series
    # v = powers_of_Mf.third * powers_of_pi["third"]
    # logv = np.log(v)
    phasing = prefactors.initial_phasing
    phasing += prefactors.two_thirds * powers_of_Mf.two_thirds
    phasing += prefactors.third * powers_of_Mf.third
    phasing += prefactors.third_with_logv * powers_of_Mf.logv * powers_of_Mf.third
    phasing += prefactors.logv * powers_of_Mf.logv
    phasing += prefactors.minus_third * powers_of_Mf.m_third
    phasing += prefactors.minus_two_thirds * powers_of_Mf.m_two_thirds
    phasing += prefactors.minus_one * powers_of_Mf.inv
    phasing += prefactors.minus_four_thirds / powers_of_Mf.four_thirds
    phasing += prefactors.minus_five_thirds * powers_of_Mf.m_five_thirds
    phasing += (prefactors.one * Mf + prefactors.four_thirds * powers_of_Mf.four_thirds +
                prefactors.five_thirds * powers_of_Mf.five_thirds +
                prefactors.two * powers_of_Mf.two) * p.etaInv
    return phasing

def init_phi_ins_prefactors(prefactors: PhiInsPrefactors, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries) -> int:
    if p is None:
        raise ValueError("p is None")
    if prefactors is None:
        raise ValueError("prefactors is None")
    sigma1 = p.sigma1
    sigma2 = p.sigma2
    sigma3 = p.sigma3
    sigma4 = p.sigma4

    prefactors.initial_phasing = pn.v[5] - LAL_PI_4
    prefactors.two_thirds = pn.v[7] * powers_of_pi["two_thirds"]
    prefactors.third = pn.v[6] * powers_of_pi["third"]
    prefactors.third_with_logv = pn.vlogv[6] * powers_of_pi["third"]
    prefactors.logv = pn.vlogv[5]
    prefactors.minus_third = pn.v[4] * powers_of_pi["m_third"]
    prefactors.minus_two_thirds = pn.v[3] * powers_of_pi["m_two_thirds"]
    prefactors.minus_one = pn.v[2] * powers_of_pi["inv"]
    prefactors.minus_four_thirds = pn.v[1] / powers_of_pi["four_thirds"]
    prefactors.minus_five_thirds = pn.v[0] * powers_of_pi["m_five_thirds"]

    prefactors.one = sigma1
    prefactors.four_thirds = sigma2 * 0.75
    prefactors.five_thirds = sigma3 * 0.6
    prefactors.two = sigma4 * 0.5

    return 0

def DPhiInsAnsatzInt(Mf: float, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries) -> float:
    sigma1 = p.sigma1
    sigma2 = p.sigma2
    sigma3 = p.sigma3
    sigma4 = p.sigma4
    Pi = LAL_PI

    v = (Pi * Mf)**(1/3)  # cbrt(Pi*Mf)
    logv = math.log(v)
    v2 = v * v
    v3 = v * v2
    v4 = v * v3
    v5 = v * v4
    v6 = v * v5
    v7 = v * v6
    v8 = v * v7

    Dphasing = 2.0 * pn.v[7] * v7
    Dphasing += (pn.v[6] + pn.vlogv[6] * (1.0 + logv)) * v6
    Dphasing += pn.vlogv[5] * v5
    Dphasing += -1.0 * pn.v[4] * v4
    Dphasing += -2.0 * pn.v[3] * v3
    Dphasing += -3.0 * pn.v[2] * v2
    Dphasing += -4.0 * pn.v[1] * v
    Dphasing += -5.0 * pn.v[0]
    Dphasing /= (v8 * 3.0)
    Dphasing *= Pi

    Dphasing += (sigma1 + sigma2 * v * powers_of_pi["m_third"] +
                 sigma3 * v2 * powers_of_pi["m_two_thirds"] +
                 (sigma4 * powers_of_pi["inv"]) * v3) * p.etaInv

    return Dphasing

# ================= Phase: Glueing functions =================

def ComputeIMRPhenomDPhaseCoefficients(p: IMRPhenomDPhaseCoefficients, eta: float, chi1: float, chi2: float, finspin: float, extraParams) -> None:
    """
    Populate all IMRPhenomD phase coefficients in ``p``.

    ``extraParams`` carries optional non-GR correction parameters.
    """
    p.eta = eta
    p.etaInv = 1.0 / eta
    p.chi1 = chi1
    p.chi2 = chi2
    eta2 = eta * eta
    p.eta2 = eta2
    p.Seta = math.sqrt(1.0 - 4.0 * eta)

    p.q = 0.5 * (1.0 + p.Seta - 2.0 * eta) * p.etaInv
    p.chi = chiPN(p.Seta, eta, chi1, chi2)
    xi = -1.0 + p.chi

    p.sigma1 = sigma1Fit(eta, eta2, xi)
    p.sigma2 = sigma2Fit(eta, eta2, xi)
    p.sigma3 = sigma3Fit(eta, eta2, xi)
    p.sigma4 = sigma4Fit(eta, eta2, xi)

    p.beta1 = beta1Fit(eta, eta2, xi)
    p.beta2 = beta2Fit(eta, eta2, xi)
    p.beta3 = beta3Fit(eta, eta2, xi)

    p.alpha1 = alpha1Fit(eta, eta2, xi)
    p.alpha2 = alpha2Fit(eta, eta2, xi)
    p.alpha3 = alpha3Fit(eta, eta2, xi)
    p.alpha4 = alpha4Fit(eta, eta2, xi)
    p.alpha5 = alpha5Fit(eta, eta2, xi)

    p.fRD = fring(eta, chi1, chi2, finspin)
    p.fDM = fdamp(eta, chi1, chi2, finspin)

def ComputeIMRPhenDPhaseConnectionCoefficients(p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries, prefactors: PhiInsPrefactors, Rholm: float, Taulm: float) -> None:
    """
    Compute connection coefficients that make the phase C1-continuous.

    ``p`` contains the phenomenological phase coefficients, ``pn`` contains
    the PN series, and ``prefactors`` contains the inspiral phase prefactors.
    """
    etaInv = p.etaInv

#   // Transition frequencies
#   // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    # PHI_fJoin_INS joins the inspiral and intermediate phase regions.
    p.fInsJoin = PHI_fJoin_INS
    p.fMRDJoin = 0.5 * p.fRD

#   // Compute C1Int and C2Int coeffs
#   // Equations to solve for to get C(1) continuous join
#   // PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
#   // Joining at fInsJoin
#   // PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
#   // PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
    DPhiIns = DPhiInsAnsatzInt(PHI_fJoin_INS, p, pn)
    DPhiInt = DPhiIntAnsatz(PHI_fJoin_INS, p)
    p.C2Int = DPhiIns - DPhiInt

    powers_of_fInsJoin = init_useful_powers(PHI_fJoin_INS)
    p.C1Int = (PhiInsAnsatzInt(PHI_fJoin_INS, powers_of_fInsJoin, prefactors, p, pn)
               - etaInv * PhiIntAnsatz(PHI_fJoin_INS, p)
               - p.C2Int * PHI_fJoin_INS)

#   // Compute C1MRD and C2MRD coeffs
#   // Equations to solve for to get C(1) continuous join
#   // PhiInsInt (f)  =   PhiMRD (f) + C1MRD + C2MRD f
#   // Joining at fMRDJoin
#   // Where \[Phi]InsInt(f) is the \[Phi]Ins+\[Phi]Int joined function
#   // PhiInsInt (fMRDJoin)  =   PhiMRD (fMRDJoin) + C1MRD + C2MRD fMRDJoin
#   // PhiInsInt'(fMRDJoin)  =   PhiMRD'(fMRDJoin) + C2MRD
#   // temporary Intermediate Phase function to Join up the Merger-Ringdown
    PhiIntTempVal = etaInv * PhiIntAnsatz(p.fMRDJoin, p) + p.C1Int + p.C2Int * p.fMRDJoin
    DPhiIntTempVal = DPhiIntTemp(p.fMRDJoin, p)
    DPhiMRDVal = DPhiMRD(p.fMRDJoin, p, Rholm, Taulm)
    p.C2MRD = DPhiIntTempVal - DPhiMRDVal
    p.C1MRD = PhiIntTempVal - etaInv * PhiMRDAnsatzInt(p.fMRDJoin, p, Rholm, Taulm) - p.C2MRD * p.fMRDJoin

def IMRPhenDPhase(f: float, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries, powers_of_f: UsefulPowers, prefactors: PhiInsPrefactors, Rholm: float, Taulm: float) -> float:
    """Evaluate the complete IMR phase in the frequency region containing ``f``."""
    if not step_func_boolean(f, p.fInsJoin):  # Inspiral region
        PhiIns = PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn)
        return PhiIns
    if step_func_boolean(f, p.fMRDJoin):  # Merger-ringdown region
        PhiMRD = p.etaInv * PhiMRDAnsatzInt(f, p, Rholm, Taulm) + p.C1MRD + p.C2MRD * f
        return PhiMRD
    PhiInt = p.etaInv * PhiIntAnsatz(f, p) + p.C1Int + p.C2Int * f
    return PhiInt

def IMRPhenDPhase_vectorized(f: np.ndarray, p: IMRPhenomDPhaseCoefficients, pn: PNPhasingSeries, prefactors: PhiInsPrefactors, Rholm: float, Taulm: float) -> np.ndarray:
    freqins, freqint, freqmrd = split_freqs_phase(f, p)
    PhiIns, PhiInt, PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
    if len(freqins) > 0:
        powers_of_freqins = init_useful_power_arrays(freqins)
        PhiIns = PhiInsAnsatzInt_vectorized(freqins, powers_of_freqins, prefactors, p, pn)
    if len(freqint) > 0:
        PhiInt = p.etaInv * PhiIntAnsatz_vectorized(freqint, p) + p.C1Int + p.C2Int * freqint
    if len(freqmrd) > 0:
        PhiMRD = p.etaInv * PhiMRDAnsatzInt_vectorized(freqmrd, p, Rholm, Taulm) + p.C1MRD + p.C2MRD * freqmrd
    return np.concatenate((PhiIns, PhiInt, PhiMRD))

def Subtract3PNSS(m1: float, m2: float, M: float, eta: float, chi1: float, chi2: float) -> float:
    m1M = m1 / M
    m2M = m2 / M
    pn_ss3 = ((326.75/1.12) + (557.5/1.8) * eta) * eta * chi1 * chi2
    pn_ss3 += (((4703.5/8.4) + (2935.0/6.0) * m1M - 120.0 * m1M * m1M) +
               ((-4108.25/6.72) - (108.5/1.2) * m1M + (125.5/3.6) * m1M * m1M)) * m1M * m1M * chi1 * chi1
    pn_ss3 += (((4703.5/8.4) + (2935.0/6.0) * m2M - 120.0 * m2M * m2M) +
               ((-4108.25/6.72) - (108.5/1.2) * m2M + (125.5/3.6) * m2M * m2M)) * m2M * m2M * chi2 * chi2
    return pn_ss3

# Core frequency-domain generator
def IMRPhenomDGenerateFD(freqs_in: np.ndarray, deltaF: float, phi0: float, fRef: float,
                          m1: float, m2: float, chi1_in: float, chi2_in: float,
                          distance: float, extraParams: dict, NRTidal_version: str) -> np.ndarray:
    """
    Generate an IMRPhenomD frequency-domain waveform and diagnostic arrays.

    ``freqs_in`` contains frequencies in Hz. A positive ``deltaF`` selects a
    uniform grid; otherwise the supplied frequencies must be strictly
    increasing. ``phi0`` is the phase at reference frequency ``fRef``;
    ``fRef=0`` selects the minimum frequency. Component masses are in solar
    masses, distance is in metres, and aligned spins are dimensionless.
    """
    # Validate physical parameters and the frequency grid.
    if freqs_in is None or len(freqs_in) == 0:
        raise ValueError("freqs_in must be a non-empty array.")
    if deltaF <= 0 and not np.all(np.diff(freqs_in) > 0):
        raise ValueError("For non-uniform frequency grid, freqs_in must be strictly increasing.")
    if phi0 is None:
        raise ValueError("phi0 must be provided.")
    if m1 <= 0 or m2 <= 0:
        raise ValueError("m1 and m2 must be positive.")
    if distance <= 0:
        raise ValueError("distance must be positive.")
    if not (-1.0 <= chi1_in <= 1.0 and -1.0 <= chi2_in <= 1.0):
        raise ValueError("Spin values must be in [-1,1].")

    f_min = freqs_in[0]
    f_max = freqs_in[-1]
    if f_min <= 0:
        raise ValueError("Minimum frequency must be positive.")
    if f_max < 0:
        raise ValueError("Maximum frequency must be non-negative.")

    q = m1 / m2 if m1 > m2 else m2 / m1
    if q > MAX_ALLOWED_MASS_RATIO:
        print(f"Warning: mass ratio q={q} exceeds maximum allowed {MAX_ALLOWED_MASS_RATIO}.")

    # A zero reference frequency selects the lower frequency bound.
    fRef_use = f_min if fRef == 0.0 else fRef

    # Total mass and symmetric mass ratio.
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    if eta > 0.25:
        eta = 0.25  # Nudge roundoff at the equal-mass limit.
    if eta < 0.0 or eta > 0.25:
        raise ValueError("Unphysical eta; must be between 0 and 0.25.")

    # Convert M to seconds for dimensionless frequencies Mf.
    M_sec = M * LAL_MTSUN_SI

    # Overall amplitude prefactor.
    amp0 = 2.0 * math.sqrt(5.0 / (64.0 * LAL_PI)) * M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance

    # Construct the working frequency grid and output offset.
    if deltaF > 0:
        npts = next_pow2(f_max / deltaF) + 1
        # The uniform output grid begins at zero frequency.
        freqs_uniform = np.linspace(0, deltaF*(npts-1), npts)
        iStart = int(f_min / deltaF)
        iStop = int(f_max / deltaF)
        offset = iStart
        freqs_used = freqs_uniform[iStart:iStop]
    else:
        npts = len(freqs_in)
        offset = 0
        freqs_used = freqs_in

    # Initialize zero-padded output arrays.
    waveform = np.zeros(npts, dtype=complex)
    amp_array = np.zeros(npts, dtype=float)
    phi_array = np.zeros(npts, dtype=float)

    # Compute IMR model coefficients.
    finspin = final_spin0815(eta, chi1_in, chi2_in)
    if finspin < MIN_FINAL_SPIN:
        print(f"Warning: final spin ({finspin}) is small; model may misbehave.")

    pAmp = IMRPhenomDAmplitudeCoefficients()
    ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1_in, chi2_in, finspin)

    if extraParams is None:
        extraParams = {}
    # IMRPhenomD uses aligned-spin phasing through 3.5PN.
    extraParams["PNSpinOrder"] = "35PN"

    pPhi = IMRPhenomDPhaseCoefficients()
    ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1_in, chi2_in, finspin, extraParams)

    # Compute the PN phase series.
    pn = XLALSimInspiralTaylorF2AlignedPhasing(m1, m2, chi1_in, chi2_in, extraParams)
    if pn is None:
        raise RuntimeError("Failed to compute PN phasing series.")

    # The 3PN spin-spin subtraction is already included in the PN coefficients.
    phi_prefactors = PhiInsPrefactors()
    status = init_phi_ins_prefactors(phi_prefactors, pPhi, pn)
    if status != 0:
        raise RuntimeError("init_phi_ins_prefactors failed.")

    ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, phi_prefactors, 1.0, 1.0)
#   //time shift so that peak amplitude is approximately at t=0
#   //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timedomain
    t0 = DPhiMRD(pAmp.fmaxCalc, pPhi, 1.0, 1.0)

    amp_prefactors = AmpInsPrefactors()
    status = init_amp_ins_prefactors(amp_prefactors, pAmp)
    if status != 0:
        raise RuntimeError("init_amp_ins_prefactors failed.")

    MfRef = M_sec * fRef_use
    powers_of_fRef = init_useful_powers(MfRef)
    phifRef = IMRPhenDPhase(MfRef, pPhi, pn, powers_of_fRef, phi_prefactors, 1.0, 1.0)
    phi_precalc = 2.0 * phi0 + phifRef

    # Generate the waveform with the vectorized production path.
    # Vectorized evaluation.
    Mf = M_sec * freqs_used
    powers_of_f = init_useful_power_arrays(Mf)
    amp = IMRPhenDAmplitude_vectorized(Mf, pAmp, powers_of_f, amp_prefactors)
    phi = IMRPhenDPhase_vectorized(Mf, pPhi, pn, phi_prefactors, 1.0, 1.0)
    phi = phi - t0 * (Mf - MfRef) - phi_precalc
    waveform[offset:offset + len(freqs_used)] = amp0 * amp * np.exp(-1j * phi)
    amp_array[offset:offset + len(freqs_used)] = amp0 * amp
    phi_array[offset:offset + len(freqs_used)] = phi


    finsjoin = PHI_fJoin_INS / M_sec
    fmrdjoin = pPhi.fMRDJoin / M_sec
    return waveform, amp_array, phi_array, finsjoin, fmrdjoin

# Public frequency-sequence wrapper
def XLALSimIMRPhenomDFrequencySequence(phi0: float, fRef_in: float, deltaF: float,
                                        m1: float, m2: float,
                                        chi1: float, chi2: float,
                                        distance_mpc: float,
                                        freqs: list[float],
                                        extraParams, NRTidal_version) -> np.ndarray:
    """
    Return the complex IMRPhenomD frequency-domain plus polarization.

    ``phi0`` is the orbital phase at ``fRef_in``; zero selects the lower
    frequency bound. ``deltaF`` is the positive frequency spacing in Hz.
    Component masses are in solar masses, distance is in Mpc, and aligned
    spins lie in [-1, 1]. ``freqs`` is ``[f_min, f_max]`` in Hz.
    """
    # Validate the public inputs.
    if deltaF <= 0:
        raise ValueError("deltaF must be positive")
    if m1 <= 0 or m2 <= 0:
        raise ValueError("m1 and m2 must be positive")
    if freqs[0] <= 0:
        raise ValueError("f_min must be positive")
    if freqs[1] < 0:
        raise ValueError("f_max must be non-negative")
    if distance_mpc <= 0:
        raise ValueError("distance must be positive")
    if not (-1.0 <= chi1 <= 1.0 and -1.0 <= chi2 <= 1.0):
        raise ValueError("Spin values must be in [-1,1]")

    # Convert luminosity distance from Mpc to metres.
    distance = distance_mpc * LAL_MPC_SI

    # Warn outside the calibrated mass-ratio range.
    q = m1 / m2 if m1 > m2 else m2 / m1
    if q > MAX_ALLOWED_MASS_RATIO:
        print("Warning: The model is not supported for high mass ratio (q > {}), but continue.".format(MAX_ALLOWED_MASS_RATIO))

    # A zero reference frequency selects f_min.
    f_min = freqs[0]
    f_max = freqs[1]
    fRef = f_min if fRef_in == 0.0 else fRef_in

    # Convert total mass to seconds for dimensionless frequencies.
    M_sec = (m1 + m2) * LAL_MTSUN_SI
    # Convert the dimensionless cutoff Mf to Hz.
    fCut = f_CUT / M_sec
    if fCut <= f_min:
        raise ValueError(f"(fCut = {fCut} Hz) <= f_min = {f_min} Hz")

    # A zero upper bound selects the model cutoff.
    f_max_prime = f_max if f_max != 0 else fCut
    f_max_prime = min(f_max_prime, fCut)
    if f_max_prime <= f_min:
        raise ValueError("f_max_prime <= f_min")

    # The core generator constructs the points between these bounds.
    freqs_boundary = [f_min, f_max_prime]

    waveform, amp_array, phi_array, finsjoin, fmrdjoin = IMRPhenomDGenerateFD(freqs_boundary, deltaF, phi0, fRef,
                                          m1, m2, chi1, chi2, distance,
                                          extraParams, NRTidal_version)

    # Zero-pad to a power-of-two-plus-one grid when f_max exceeds the cutoff.
    if f_max_prime < f_max:
        n_full = next_pow2(f_max / deltaF) + 1
        waveform_full = np.zeros(n_full, dtype=complex)
        n_current = waveform.size
        waveform_full[:n_current] = waveform
        waveform = waveform_full

    return waveform

def IMRPhenomDPhaseFrequencySequence(phases: np.ndarray,
                                      freqs: np.ndarray,
                                      ind_min: int,
                                      ind_max: int,
                                      m1: float,
                                      m2: float,
                                      chi1x: float,
                                      chi1y: float,
                                      chi1z: float,
                                      chi2x: float,
                                      chi2y: float,
                                      chi2z: float,
                                      Rholm: float,
                                      Taulm: float,
                                      extraParams: dict) -> np.ndarray:
    """
    Fill ``phases[ind_min:ind_max]`` with the IMRPhenomD phase in radians.

    ``freqs`` contains dimensionless geometric frequencies Mf. Component
    masses are in solar masses; the spin arguments provide Cartesian
    components. ``Rholm`` and ``Taulm`` are QNM correction factors used by
    higher-mode extensions. The supplied ``phases`` array is returned.
    """
    # Precompute the amplitude and phase structures.
    pD = IMRPhenomDSetupAmpAndPhaseCoefficients(m1, m2,
                                                 chi1x, chi1y, chi1z,
                                                 chi2x, chi2y, chi2z,
                                                 Rholm, Taulm,
                                                 extraParams)
    for i in range(ind_min, ind_max):
        Mf = freqs[i]
        powers_of_f = init_useful_powers(Mf)
        phase_val = IMRPhenDPhase(Mf, pD.pPhi, pD.pn, powers_of_f, pD.phi_prefactors, Rholm, Taulm)
        phases[i] = phase_val
    return phases

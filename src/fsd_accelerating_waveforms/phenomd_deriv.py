from .phenomd import *

def DPhiMRD_vectorized(f: np.ndarray, p:IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the first frequency derivative of the merger-ringdown phase. The C2MRD connection term is included.
    """
    term1 = p.alpha1
    term2 = p.alpha2 / (f**2)
    term3 = p.alpha3 / (f**0.25)
    denom = p.fDM * Taulm * (1 + ((f - p.alpha5 * p.fRD)**2) / ((p.fDM * Taulm * Rholm)**2))
    term4 = p.alpha4 / denom
    return (term1 + term2 + term3 + term4) * p.etaInv + p.C2MRD

def D2PhiMRD_vectorized(f: np.ndarray, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the second frequency derivative of the merger-ringdown phase.
    """
    term1 = -2 * p.alpha2 / f**3
    term2 = -p.alpha3 / (4 * f**(5/4))
    # The ringdown and damping frequencies are stored on the coefficients.
    term3 = -2 * p.alpha4 * (f - p.alpha5 * p.fRD) / (
                p.fDM**3 * Rholm**2 * Taulm**3 *
                (((f - p.alpha5 * p.fRD)**2) / (p.fDM**2 * Rholm**2 * Taulm**2) + 1)**2
            )
    return p.etaInv * (term1 + term2 + term3)

def D3PhiMRD_vectorized(f: np.ndarray, p:IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the third frequency derivative of the merger-ringdown phase.
    """
    term1 = 6 * p.alpha2 / f**4
    term2 = 5 * p.alpha3 / (16 * f**(9/4))
    shift = f - p.alpha5 * p.fRD
    common_den = (shift**2) / (p.fDM**2 * Rholm**2 * Taulm**2) + 1
    term3 = p.alpha4 * Rholm * (
                8 * shift**2 / (p.fDM**5 * Rholm**5 * Taulm**5 * common_den**3)
                - 2 / (p.fDM**3 * Rholm**3 * Taulm**3 * common_den**2)
            )
    return p.etaInv * (term1 + term2 + term3)

def D4PhiMRD_vectorized(f: np.ndarray, p:IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the fourth frequency derivative of the merger-ringdown phase.
    """
    term1 = -24 * p.alpha2 / f**5
    term2 = -45 * p.alpha3 / (64 * f**(13/4))
    shift = f - p.alpha5 * p.fRD
    common_den = (shift**2) / (p.fDM**2 * Rholm**2 * Taulm**2) + 1
    term3 = p.alpha4 * Rholm * (
                24 * shift / (p.fDM**5 * Rholm**5 * Taulm**5 * common_den**3)
                - 48 * shift**3 / (p.fDM**7 * Rholm**7 * Taulm**7 * common_den**4)
            )
    return p.etaInv * (term1 + term2 + term3)

def D5PhiMRD_vectorized(f: np.ndarray, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the fifth frequency derivative of the merger-ringdown phase.
    """
    term1 = 120 * p.alpha2 / f**6
    term2 = 585 * p.alpha3 / (256 * f**(17/4))
    shift = f - p.alpha5 * p.fRD
    common_den = (shift**2) / (p.fDM**2 * Rholm**2 * Taulm**2) + 1
    term3 = p.alpha4 * Rholm * (
                384 * shift**4 / (p.fDM**9 * Rholm**9 * Taulm**9 * common_den**5)
                - 288 * shift**2 / (p.fDM**7 * Rholm**7 * Taulm**7 * common_den**4)
                + 24 / (p.fDM**5 * Rholm**5 * Taulm**5 * common_den**3)
            )
    return p.etaInv * (term1 + term2 + term3)

def D6PhiMRD_vectorized(f: np.ndarray, p: IMRPhenomDPhaseCoefficients, Rholm: float, Taulm: float) -> np.ndarray:
    """
    Return the sixth frequency derivative of the merger-ringdown phase.
    """
    term1 = -720 * p.alpha2 / f**7
    term2 = -9945 * p.alpha3 / (1024 * f**(21/4))
    shift = f - p.alpha5 * p.fRD
    common_den = (shift**2) / (p.fDM**2 * Rholm**2 * Taulm**2) + 1
    term3 = p.alpha4 * Rholm * (
                -3840 * shift**5 / (p.fDM**11 * Rholm**11 * Taulm**11 * common_den**6)
                + 3840 * shift**3 / (p.fDM**9 * Rholm**9 * Taulm**9 * common_den**5)
                - 720 * shift / (p.fDM**7 * Rholm**7 * Taulm**7 * common_den**4)
            )
    return p.etaInv * (term1 + term2 + term3)

def DPhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the first frequency derivative of the intermediate phase. The C2Int connection term is included.
    """
    return (p.beta1 + p.beta3 / (Mf**4) + p.beta2 / Mf) * p.etaInv + p.C2Int

def D2PhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the second frequency derivative of the intermediate phase.
    """
    return p.etaInv * (-p.beta2 / Mf**2 - 4 * p.beta3 / Mf**5)


def D3PhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the third frequency derivative of the intermediate phase.
    """
    return p.etaInv * (2 * p.beta2 / Mf**3 + 20 * p.beta3 / Mf**6)


def D4PhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the fourth frequency derivative of the intermediate phase.
    """
    return p.etaInv * (-6 * p.beta2 / Mf**4 - 120 * p.beta3 / Mf**7)


def D5PhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the fifth frequency derivative of the intermediate phase.
    """
    return p.etaInv * (24 * p.beta2 / Mf**5 + 840 * p.beta3 / Mf**8)


def D6PhiInt_vectorized(Mf: np.ndarray, p: IMRPhenomDPhaseCoefficients) -> np.ndarray:
    """
    Return the sixth frequency derivative of the intermediate phase.
    """
    return p.etaInv * (-120 * p.beta2 / Mf**6 - 6720 * p.beta3 / Mf**9)

def DPhiIns_vectorized(
    Mf: np.ndarray,
    powers_of_Mf: UsefulPowerArrays,
    p: IMRPhenomDPhaseCoefficients,
    prefactors: PhiInsPrefactors,
) -> np.ndarray:
    """
    Return the first frequency derivative of the inspiral phase, factored by Mf**-1.
    """
    # Factor out Mf**-1 from the derivative.
    outside = powers_of_Mf.inv

    # Remaining polynomial and logarithmic terms.
    A = (
        p.etaInv * (
            5 * prefactors.five_thirds / 3   * powers_of_Mf.five_thirds
          + 4 * prefactors.four_thirds / 3   * powers_of_Mf.four_thirds
          + 2 * prefactors.two               * powers_of_Mf.two
          + prefactors.one                   * Mf
        )
        - 5 * prefactors.minus_five_thirds / 3  * powers_of_Mf.m_five_thirds
        - 4 * prefactors.minus_four_thirds / 3  * (1 / powers_of_Mf.four_thirds)
        -     prefactors.minus_third / 3        * powers_of_Mf.m_third
        - 2 * prefactors.minus_two_thirds / 3   * powers_of_Mf.m_two_thirds
        +     prefactors.third / 3             * powers_of_Mf.third
        +     prefactors.third_with_logv / 3   * powers_of_Mf.third
        +     prefactors.third_with_logv / 3   * powers_of_Mf.third * powers_of_Mf.logv
        -     prefactors.minus_one            * powers_of_Mf.inv
        +     prefactors.logv / 3
        + 2 * prefactors.two_thirds / 3       * powers_of_Mf.two_thirds
    )

    return outside * A

def D2PhiIns_vectorized(Mf: np.ndarray,
                        powers_of_Mf: UsefulPowerArrays,
                        p: IMRPhenomDPhaseCoefficients,
                        prefactors: PhiInsPrefactors
                       ) -> np.ndarray:
    """
    Return the second frequency derivative of the inspiral phase, factored by Mf**-2.
    """
    # Factor out Mf**-2 from the derivative.
    outside = powers_of_Mf.inv ** 2

    # Remaining polynomial and logarithmic terms.
    A2 = (
        - prefactors.logv / 3
        + 40 * prefactors.minus_five_thirds / 9   * powers_of_Mf.m_five_thirds
        + 28 * prefactors.minus_four_thirds / 9   * (1 / powers_of_Mf.four_thirds)
        + 2  * prefactors.minus_one               * powers_of_Mf.inv
        + 4  * prefactors.minus_third  / 9        * powers_of_Mf.m_third
        + 10 * prefactors.minus_two_thirds / 9    * powers_of_Mf.m_two_thirds
        - 2  * prefactors.third        / 9        * powers_of_Mf.third
        - prefactors.third_with_logv   / 9        * powers_of_Mf.third
        + p.etaInv * (
              10 * prefactors.five_thirds / 9     * powers_of_Mf.five_thirds
            + 4  * prefactors.four_thirds / 9     * powers_of_Mf.four_thirds
            + 2  * prefactors.two                 * powers_of_Mf.two
          )
        - 2 * prefactors.two_thirds   / 9       * powers_of_Mf.two_thirds
        - 2 * prefactors.third_with_logv / 9    * powers_of_Mf.third * powers_of_Mf.logv
    )

    return outside * A2

def D3PhiIns_vectorized(Mf: np.ndarray,
                        powers_of_Mf: UsefulPowerArrays,
                        p: IMRPhenomDPhaseCoefficients,
                        prefactors: PhiInsPrefactors
                       ) -> np.ndarray:
    """
    Return the third frequency derivative of the inspiral phase, factored by Mf**-3.
    """
    outside = powers_of_Mf.inv ** 3
    A3 = (
        p.etaInv * (
            -10 * prefactors.five_thirds / 27 * powers_of_Mf.five_thirds
            -8  * prefactors.four_thirds / 27 * powers_of_Mf.four_thirds
        )
        - 440 * prefactors.minus_five_thirds / 27 * powers_of_Mf.m_five_thirds
        - 280 * prefactors.minus_four_thirds / 27 * (1 / powers_of_Mf.four_thirds)
        - 28  * prefactors.minus_third / 27       * powers_of_Mf.m_third
        - 80  * prefactors.minus_two_thirds / 27  * powers_of_Mf.m_two_thirds
        + 10  * prefactors.third / 27            * powers_of_Mf.third
        + prefactors.third_with_logv / 9         * powers_of_Mf.third
        + 10  * prefactors.third_with_logv / 27  * powers_of_Mf.third * powers_of_Mf.logv
        + 8   * prefactors.two_thirds / 27       * powers_of_Mf.two_thirds
        - 6   * prefactors.minus_one             * powers_of_Mf.inv
        + 2   * prefactors.logv / 3
    )
    return outside * A3


def D4PhiIns_vectorized(Mf: np.ndarray,
                        powers_of_Mf: UsefulPowerArrays,
                        p: IMRPhenomDPhaseCoefficients,
                        prefactors: PhiInsPrefactors
                       ) -> np.ndarray:
    """
    Return the fourth frequency derivative of the inspiral phase, factored by Mf**-4.
    """
    outside = powers_of_Mf.inv ** 4
    A4 = (
        p.etaInv * (
            40 * prefactors.five_thirds / 81 * powers_of_Mf.five_thirds
           +40 * prefactors.four_thirds / 81 * powers_of_Mf.four_thirds
        )
        + 6160 * prefactors.minus_five_thirds / 81 * powers_of_Mf.m_five_thirds
        + 3640 * prefactors.minus_four_thirds / 81 * (1 / powers_of_Mf.four_thirds)
        + 280  * prefactors.minus_third / 81       * powers_of_Mf.m_third
        + 880  * prefactors.minus_two_thirds / 81  * powers_of_Mf.m_two_thirds
        - 80   * prefactors.third / 81             * powers_of_Mf.third
        - 14   * prefactors.third_with_logv / 81   * powers_of_Mf.third
        - 80   * prefactors.third_with_logv / 81   * powers_of_Mf.third * powers_of_Mf.logv
        - 56   * prefactors.two_thirds / 81        * powers_of_Mf.two_thirds
        + 24   * prefactors.minus_one              * powers_of_Mf.inv
        - 2    * prefactors.logv
    )
    return outside * A4


def D5PhiIns_vectorized(Mf: np.ndarray,
                        powers_of_Mf: UsefulPowerArrays,
                        p: IMRPhenomDPhaseCoefficients,
                        prefactors: PhiInsPrefactors
                       ) -> np.ndarray:
    """
    Return the fifth frequency derivative of the inspiral phase, factored by Mf**-5.
    """
    outside = powers_of_Mf.inv ** 5
    A5 = (
        p.etaInv * (
            -280 * prefactors.five_thirds / 243 * powers_of_Mf.five_thirds
            -320 * prefactors.four_thirds / 243 * powers_of_Mf.four_thirds
        )
        - 104720 * prefactors.minus_five_thirds / 243 * powers_of_Mf.m_five_thirds
        - 58240  * prefactors.minus_four_thirds / 243 * (1 / powers_of_Mf.four_thirds)
        - 3640   * prefactors.minus_third / 243       * powers_of_Mf.m_third
        - 12320  * prefactors.minus_two_thirds / 243  * powers_of_Mf.m_two_thirds
        + 880    * prefactors.third / 243            * powers_of_Mf.third
        + 74     * prefactors.third_with_logv / 243   * powers_of_Mf.third
        + 880    * prefactors.third_with_logv / 243   * powers_of_Mf.third * powers_of_Mf.logv
        + 560    * prefactors.two_thirds / 243        * powers_of_Mf.two_thirds
        - 120    * prefactors.minus_one                * powers_of_Mf.inv
        + 8                                       * prefactors.logv
    )
    return outside * A5


def D6PhiIns_vectorized(Mf: np.ndarray,
                        powers_of_Mf: UsefulPowerArrays,
                        p: IMRPhenomDPhaseCoefficients,
                        prefactors: PhiInsPrefactors
                       ) -> np.ndarray:
    """
    Return the sixth frequency derivative of the inspiral phase, factored by Mf**-6.
    """
    outside = powers_of_Mf.inv ** 6
    A6 = (
        p.etaInv * (
            2800 * prefactors.five_thirds / 729 * powers_of_Mf.five_thirds
           +3520 * prefactors.four_thirds / 729 * powers_of_Mf.four_thirds
        )
        + 2094400 * prefactors.minus_five_thirds / 729 * powers_of_Mf.m_five_thirds
        + 1106560 * prefactors.minus_four_thirds / 729 * (1 / powers_of_Mf.four_thirds)
        + 58240   * prefactors.minus_third / 729       * powers_of_Mf.m_third
        + 209440  * prefactors.minus_two_thirds / 729  * powers_of_Mf.m_two_thirds
        - 12320   * prefactors.third / 729             * powers_of_Mf.third
        - 52      * prefactors.third_with_logv / 243   * powers_of_Mf.third
        - 12320   * prefactors.third_with_logv / 729   * powers_of_Mf.third * powers_of_Mf.logv
        - 7280    * prefactors.two_thirds / 729        * powers_of_Mf.two_thirds
        + 720     * prefactors.minus_one                * powers_of_Mf.inv
        - 40                                         * prefactors.logv
    )
    return outside * A6

def DAmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:

    fRD = p.fRD
    fDM = p.fDM
    gamma1 = p.gamma1
    gamma2 = p.gamma2
    gamma3 = p.gamma3

    fDMgamma3 = fDM * gamma3
    pow2_fDMgamma3 = (fDMgamma3)**2
    fminfRD = f - fRD
    expfactor = np.exp((fminfRD * gamma2) / fDMgamma3)
    pow2plus = (fminfRD)**2 + pow2_fDMgamma3

    return ((-2 * fDM * fminfRD * gamma3 * gamma1) / pow2plus - (gamma2 * gamma1)) / (expfactor * pow2plus)

def D2AmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the second frequency derivative of the merger-ringdown amplitude ansatz.
    """
    fminfRD = f - p.fRD
    fDMgamma3 = p.fDM - p.gamma3
    exp_factor = np.exp(-fminfRD * p.gamma2 / fDMgamma3)
    numerator = p.gamma1 * exp_factor * (
          fDMgamma3**4 * (p.gamma2**2 - 2)
        + 4 * fDMgamma3**3 * fminfRD * p.gamma2
        + 2 * fDMgamma3**2 * fminfRD**2 * (p.gamma2**2 + 3)
        + 4 * fDMgamma3 * fminfRD**3 * p.gamma2
        + fminfRD**4 * (p.gamma2**2)
    )
    denominator = fDMgamma3 * (fDMgamma3**2 + fminfRD**2)**3
    return numerator / denominator


def D3AmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the third frequency derivative of the merger-ringdown amplitude ansatz.
    """
    fminfRD = f - p.fRD
    fDMgamma3 = p.fDM - p.gamma3
    exp_factor = np.exp(-fminfRD * p.gamma2 / fDMgamma3)
    numerator = p.gamma1 * exp_factor * (
          fDMgamma3**6 * p.gamma2 * (p.gamma2**2 - 6)
        + 6 * fDMgamma3**5 * fminfRD * (p.gamma2**2 - 4)
        + 3 * fDMgamma3**4 * fminfRD**2 * p.gamma2 * (p.gamma2**2 + 4)
        + 12 * fDMgamma3**3 * fminfRD**3 * (p.gamma2**2 + 2)
        + 3 * fDMgamma3**2 * fminfRD**4 * p.gamma2 * (p.gamma2**2 + 6)
        + 6 * fDMgamma3 * fminfRD**5 * (p.gamma2**2)
        + fminfRD**6 * (p.gamma2**3)
    )
    # Odd-order differentiation contributes the overall minus sign.
    numerator = -numerator
    denominator = fDMgamma3**2 * (fDMgamma3**2 + fminfRD**2)**4
    return numerator / denominator


def D4AmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the fourth frequency derivative of the merger-ringdown amplitude ansatz.
    """
    fminfRD = f - p.fRD
    fDMgamma3 = p.fDM - p.gamma3
    exp_factor = np.exp(-fminfRD * p.gamma2 / fDMgamma3)
    numerator = p.gamma1 * exp_factor * (
          fDMgamma3**8 * (p.gamma2**4 - 12 * p.gamma2**2 + 24)
        + 8 * fDMgamma3**7 * fminfRD * p.gamma2 * (p.gamma2**2 - 12)
        + 4 * fDMgamma3**6 * fminfRD**2 * (p.gamma2**4 + 3 * p.gamma2**2 - 60)
        + 24 * fDMgamma3**5 * fminfRD**3 * (p.gamma2**3)
        + 6 * fDMgamma3**4 * fminfRD**4 * (p.gamma2**4 + 10 * p.gamma2**2 + 20)
        + 24 * fDMgamma3**3 * fminfRD**5 * p.gamma2 * (p.gamma2**2 + 4)
        + 4 * fDMgamma3**2 * fminfRD**6 * p.gamma2**2 * (p.gamma2**2 + 9)
        + 8 * fDMgamma3 * fminfRD**7 * p.gamma2**3
        + fminfRD**8 * p.gamma2**4
    )
    denominator = fDMgamma3**3 * (fDMgamma3**2 + fminfRD**2)**5
    return numerator / denominator


def D5AmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the fifth frequency derivative of the merger-ringdown amplitude ansatz.
    """
    fminfRD = f - p.fRD
    fDMgamma3 = p.fDM - p.gamma3
    exp_factor = np.exp(-fminfRD * p.gamma2 / fDMgamma3)
    numerator = p.gamma1 * exp_factor * (
          fDMgamma3**10 * p.gamma2 * (p.gamma2**4 - 20 * p.gamma2**2 + 120)
        + 10 * fDMgamma3**9 * fminfRD * (p.gamma2**4 - 24 * p.gamma2**2 + 72)
        + 5 * fDMgamma3**8 * fminfRD**2 * p.gamma2 * (p.gamma2**4 - 216)
        + 40 * fDMgamma3**7 * fminfRD**3 * (p.gamma2**4 - 6 * p.gamma2**2 - 60)
        + 10 * fDMgamma3**6 * fminfRD**4 * p.gamma2 * (p.gamma2**4 + 12 * p.gamma2**2 - 60)
        + 60 * fDMgamma3**5 * fminfRD**5 * (p.gamma2**4 + 4 * p.gamma2**2 + 12)
        + 10 * fDMgamma3**4 * fminfRD**6 * p.gamma2 * (p.gamma2**4 + 16 * p.gamma2**2 + 60)
        + 40 * fDMgamma3**3 * fminfRD**7 * p.gamma2**2 * (p.gamma2**2 + 6)
        + 5 * fDMgamma3**2 * fminfRD**8 * p.gamma2**3 * (p.gamma2**2 + 12)
        + 10 * fDMgamma3 * fminfRD**9 * p.gamma2**4
        + fminfRD**10 * p.gamma2**5
    )
    numerator = -numerator
    denominator = fDMgamma3**4 * (fDMgamma3**2 + fminfRD**2)**6
    return numerator / denominator


def D6AmpMRD_vectorized(f: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the sixth frequency derivative of the merger-ringdown amplitude ansatz.
    """
    fminfRD = f - p.fRD
    fDMgamma3 = p.fDM - p.gamma3
    exp_factor = np.exp(-fminfRD * p.gamma2 / fDMgamma3)
    numerator = p.gamma1 * exp_factor * (
          fDMgamma3**12 * (p.gamma2**6 - 30 * p.gamma2**4 + 360 * p.gamma2**2 - 720)
        + 12 * fDMgamma3**11 * fminfRD * p.gamma2 * (p.gamma2**4 - 40 * p.gamma2**2 + 360)
        + 6 * fDMgamma3**10 * fminfRD**2 * (p.gamma2**6 - 5 * p.gamma2**4 - 480 * p.gamma2**2 + 2520)
        + 60 * fDMgamma3**9 * fminfRD**3 * p.gamma2 * (p.gamma2**4 - 16 * p.gamma2**2 - 168)
        + 15 * fDMgamma3**8 * fminfRD**4 * (p.gamma2**6 + 12 * p.gamma2**4 - 336 * p.gamma2**2 - 1680)
        + 120 * fDMgamma3**7 * fminfRD**5 * p.gamma2 * (p.gamma2**4 - 84)
        + 20 * fDMgamma3**6 * fminfRD**6 * (p.gamma2**6 + 21 * p.gamma2**4 + 252)
        + 120 * fDMgamma3**5 * fminfRD**7 * p.gamma2 * (p.gamma2**4 + 8 * p.gamma2**2 + 36)
        + 15 * fDMgamma3**4 * fminfRD**8 * p.gamma2**2 * (p.gamma2**4 + 22 * p.gamma2**2 + 120)
        + 60 * fDMgamma3**3 * fminfRD**9 * p.gamma2**3 * (p.gamma2**2 + 8)
        + 6 * fDMgamma3**2 * fminfRD**10 * p.gamma2**4 * (p.gamma2**2 + 15)
        + 12 * fDMgamma3 * fminfRD**11 * p.gamma2**5
        + fminfRD**12 * (p.gamma2**6)
    )
    denominator = fDMgamma3**5 * (fDMgamma3**2 + fminfRD**2)**7
    return numerator / denominator

def DAmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Vectorized version of the function to calculate the amplitude derivative with respect to frequency.
    """
    return p.delta1 + 2 * p.delta2 * Mf + 3 * p.delta3 * Mf**2 + 4 * p.delta4 * Mf**3

def D2AmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the second frequency derivative of the intermediate amplitude ansatz.
    """
    return 2 * p.delta2 + 6 * p.delta3 * Mf + 12 * p.delta4 * Mf**2

def D3AmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the third frequency derivative of the intermediate amplitude ansatz.
    """
    return 6 * p.delta3 + 24 * p.delta4 * Mf

def D4AmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the fourth frequency derivative of the intermediate amplitude ansatz.
    """
    return 24 * p.delta4 * np.ones_like(Mf)

def D5AmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the fifth frequency derivative of the intermediate amplitude ansatz.
    """
    return np.zeros_like(Mf)  # The fourth derivative is constant.

def D6AmpInt_vectorized(Mf: np.ndarray, p: IMRPhenomDAmplitudeCoefficients) -> np.ndarray:
    """
    Return the sixth frequency derivative of the intermediate amplitude ansatz.
    """
    return np.zeros_like(Mf)  # All higher derivatives vanish.

def DAmpIns_vectorized(
    Mf: np.ndarray,
    powers_of_Mf: UsefulPowerArrays,
    prefactors: AmpInsPrefactors,
) -> np.ndarray:
    """
    Inspiral amplitude first-frequency derivative.
    Expression:
      (8/3)*eightThirds * f**(5/3)
    + (5/3)*fiveThirds  * f**(2/3)
    + (7/3)*sevenThirds * f**(4/3)
    + 3*three           * f**2
    + (4/3)*fourThirds  * f**(1/3)
    + 2*two             * f
    + (2/3)*twoThirds   * f**(-?) actually sqrt[3]{f}^{-1}*f^{2/3} = f^{1/3}
    + one
    """
    return (
        8/3 * prefactors.eight_thirds   * powers_of_Mf.five_thirds
      + 5/3 * prefactors.five_thirds    * powers_of_Mf.two_thirds
      + 7/3 * prefactors.seven_thirds   * powers_of_Mf.four_thirds
      + 3   * prefactors.three          * powers_of_Mf.two
      + 4/3 * prefactors.four_thirds   * powers_of_Mf.third
      + 2   * prefactors.two            * Mf
      + 2/3 * prefactors.two_thirds     * powers_of_Mf.third
      +       prefactors.one
    )


def D2AmpIns_vectorized(Mf: np.ndarray,
                         powers_of_Mf: UsefulPowerArrays,
                         prefactors: AmpInsPrefactors
                        ) -> np.ndarray:
    """
    Inspiral amplitude second-frequency derivative.
    Expression:
      (40/9)*eightThirds * f**(2/3)
    + (4/9)*fourThirds  * f**(-2/3)
    - (2/9)*twoThirds   * f**(-4/3)
    + (10/9)*fiveThirds * f**(1/3)
    + (28/9)*sevenThirds* f**(1/3)
    + 6*three           * f
    + 2*two
    """
    return (
        40/9 * prefactors.eight_thirds    * powers_of_Mf.two_thirds
      + 4/9  * prefactors.four_thirds    * (1 / powers_of_Mf.two_thirds)
      - 2/9  * prefactors.two_thirds     * (1 / powers_of_Mf.four_thirds)
      + 10/9 * prefactors.five_thirds    * powers_of_Mf.third
      + 28/9 * prefactors.seven_thirds   * powers_of_Mf.third
      + 6    * prefactors.three          * Mf
      + 2    * prefactors.two
    )


def D3AmpIns_vectorized(Mf: np.ndarray,
                         powers_of_Mf: UsefulPowerArrays,
                         prefactors: AmpInsPrefactors
                        ) -> np.ndarray:
    """
    Inspiral amplitude third-frequency derivative.
    Expression:
      (80/27)*eightThirds * f**(-1/3)
    - (10/27)*fiveThirds  * f**(-4/3)
    - (8/27)*fourThirds  * f**(-5/3)
    + (28/27)*sevenThirds* f**(-2/3)
    + (8/27)*twoThirds   * f**(-7/3)
    + 6*three
    """
    return (
        80/27 * prefactors.eight_thirds    * (1 / powers_of_Mf.third)
      - 10/27 * prefactors.five_thirds     * (1 / powers_of_Mf.four_thirds)
      - 8/27  * prefactors.four_thirds     * (1 / powers_of_Mf.five_thirds)
      + 28/27 * prefactors.seven_thirds    * (1 / powers_of_Mf.two_thirds)
      + 8/27  * prefactors.two_thirds     * (1 / (powers_of_Mf.two * powers_of_Mf.third))
      + 6     * prefactors.three
    )


def D4AmpIns_vectorized(Mf: np.ndarray,
                         powers_of_Mf: UsefulPowerArrays,
                         prefactors: AmpInsPrefactors
                        ) -> np.ndarray:
    """
    Inspiral amplitude fourth-frequency derivative.
    Expression:
      -(80/81)*eightThirds * f**(-4/3)
    + (40/81)*fiveThirds  * f**(-7/3)
    + (40/81)*fourThirds  * f**(-8/3)
    - (56/81)*sevenThirds * f**(-5/3)
    - (56/81)*twoThirds   * f**(-10/3)
    """
    return (
        -80/81 * prefactors.eight_thirds   * (1 / powers_of_Mf.four_thirds)
      + 40/81 * prefactors.five_thirds    * (1 / (powers_of_Mf.two * powers_of_Mf.third))
      + 40/81 * prefactors.four_thirds    * (1 / (powers_of_Mf.two * powers_of_Mf.two_thirds))
      - 56/81 * prefactors.seven_thirds   * (1 / powers_of_Mf.five_thirds)
      - 56/81 * prefactors.two_thirds     * (1 / (powers_of_Mf.two**2 * powers_of_Mf.third))
    )


def D5AmpIns_vectorized(Mf: np.ndarray,
                         powers_of_Mf: UsefulPowerArrays,
                         prefactors: AmpInsPrefactors
                        ) -> np.ndarray:
    """
    Inspiral amplitude fifth-frequency derivative.
    Expression:
      (320/243)*eightThirds * f**(-7/3)
    - (280/243)*fiveThirds  * f**(-10/3)
    - (320/243)*fourThirds  * f**(-11/3)
    + (280/243)*sevenThirds * f**(-8/3)
    + (560/243)*twoThirds   * f**(-13/3)
    """
    return (
        320/243 * prefactors.eight_thirds  * (1 / (powers_of_Mf.two * powers_of_Mf.third))
      - 280/243 * prefactors.five_thirds   * (1 / powers_of_Mf.four_thirds)
      - 320/243 * prefactors.four_thirds   * (1 / powers_of_Mf.five_thirds)
      + 280/243 * prefactors.seven_thirds  * (1 / (powers_of_Mf.two * powers_of_Mf.two_thirds))
      + 560/243 * prefactors.two_thirds    * (1 / (powers_of_Mf.two**2 * powers_of_Mf.third))
    )


def D6AmpIns_vectorized(Mf: np.ndarray,
                         powers_of_Mf: UsefulPowerArrays,
                         prefactors: AmpInsPrefactors
                        ) -> np.ndarray:
    """
    Inspiral amplitude sixth-frequency derivative.
    Expression:
      -(2240/729)*eightThirds * f**(-10/3)
    + (2800/729)*fiveThirds  * f**(-13/3)
    + (3520/729)*fourThirds  * f**(-14/3)
    - (2240/729)*sevenThirds * f**(-11/3)
    - (7280/729)*twoThirds   * f**(-16/3)
    """
    return (
        -2240/729 * prefactors.eight_thirds  * (1 / powers_of_Mf.four_thirds)
      + 2800/729 * prefactors.five_thirds   * (1 / (powers_of_Mf.two * powers_of_Mf.third))
      + 3520/729 * prefactors.four_thirds   * (1 / (powers_of_Mf.two * powers_of_Mf.two_thirds))
      - 2240/729 * prefactors.seven_thirds  * (1 / powers_of_Mf.five_thirds)
      - 7280/729 * prefactors.two_thirds    * (1 / (powers_of_Mf.two**2 * powers_of_Mf.third))
    )

def waveform_deriv0_vector(amp, phase, freq):
    """Return the undifferentiated frequency-domain waveform."""
    return freq**(-7/6) * amp[0] * np.exp(-1j * phase[0])

def waveform_deriv1_vector(amp_derivs, phase_derivs, freq):
    """Return the first frequency derivative of the waveform."""
    amp, Damp = amp_derivs[0], amp_derivs[1]
    phase, Dphase = phase_derivs[0], phase_derivs[1]

    i = 1j
    numerator = (6 * freq * Damp + amp * (-7 - 6 * i * freq * Dphase))
    denominator = 6 * freq**(13/6)
    amp_1order = numerator / denominator
    return amp_1order * np.exp(-1j * phase)

def waveform_deriv2_vector(amp_derivs, phase_derivs, freq):
    """Return the second frequency derivative of the waveform."""
    amp, Damp, D2amp = amp_derivs[0:3]
    phase, Dphase, D2phase = phase_derivs[0:3]
    term1_part1 = -3 * freq * D2amp
    term1_part2 = Damp * (7 + 6j * freq * Dphase)
    term1 = 12 * freq * (term1_part1 + term1_part2)

    term2 = amp * (
        36j * freq**2 * D2phase +
        36 * freq**2 * Dphase**2 -
        84j * freq * Dphase -
        91
    )

    numerator = term1 + term2
    denominator = 36 * freq**(19/6)
    amp_2order = - numerator / denominator
    return amp_2order*np.exp(-1j*phase)

def waveform_deriv3_vector(amp_derivs, phase_derivs, freq):
    """Return the third frequency derivative of the waveform."""
    amp, Damp, D2amp, D3amp = amp_derivs[0:4]
    phase, Dphase, D2phase, D3phase = phase_derivs[0:4]
    i = 1j
    # Terms proportional to the undifferentiated amplitude.
    T1 = amp * (
            -216 * i * freq**3 * D3phase
            + 216 * i * freq**3 * Dphase**3
            + 756 * i * freq**2 * D2phase
            + 756 * freq**2 * Dphase**2
            - 18 * freq * Dphase * (36 * freq**2 * D2phase + 91 * i)
            - 1729
        )
    # Terms containing amplitude derivatives.
    T2 = -18 * freq * (
            Damp * (36 * i * freq**2 * D2phase + 36 * freq**2 * Dphase**2 - 84 * i * freq * Dphase - 91)
            + 6 * freq * (-2 * freq * D3amp + D2amp * (7 + 6 * i * freq * Dphase))
        )
    numerator = (T1 + T2)
    denominator = 216 * freq**(25/6)
    amp_3order = numerator / denominator
    return amp_3order * np.exp(-1j * phase)

def waveform_deriv4_vector(amp_derivs, phase_derivs, f):
    """Return the fourth frequency derivative of the waveform."""
    amp, Damp, D2amp, D3amp, D4amp = amp_derivs[0:5]
    phase, Dphase, D2phase, D3phase, D4phase = phase_derivs[0:5]
    i = 1j
    # Terms containing amplitude derivatives.
    term_A = Damp * (
        -216 * f**3 * D3phase + 216 * f**3 * Dphase**3 + 756 * f**2 * D2phase
        - 756 * i * f**2 * Dphase**2
        + 18 * i * f * Dphase * (36 * f**2 * D2phase + 91 * i)
        + 1729 * i
    )
    term_B = 9 * i * f * (
        -6 * f**2 * D4amp + 4 * f * D3amp * (7 + 6 * i * f * Dphase)
        + D2amp * (36 * i * f**2 * D2phase + 36 * f**2 * Dphase**2 - 84 * i * f * Dphase - 91)
    )
    T1 = 24 * i * f * (term_A + term_B)

    # Terms proportional to the undifferentiated amplitude.
    T2 = amp * (
        -1296 * i * f**4 * D4phase - 3888 * f**4 * D2phase**2 + 1296 * f**4 * Dphase**4
        + 6048 * i * f**3 * D3phase - 6048 * i * f**3 * Dphase**3 - 19656 * i * f**2 * D2phase
        + 216 * i * f**2 * Dphase**2 * (36 * f**2 * D2phase + 91 * i)
        - 24 * f * Dphase * (216 * f**3 * D3phase - 756 * f**2 * D2phase - 1729 * i)
        + 43225
    )

    numerator = (T1 + T2)
    denominator = 1296 * f**(31/6)

    amp_4order = numerator / denominator
    return amp_4order * np.exp(-1j * phase)

def waveform_deriv5_vector(amp_derivs, phase_derivs, freq):
    """Return the fifth frequency derivative of the waveform."""
    amp, Damp, D2amp, D3amp, D4amp, D5amp = amp_derivs[0:6]
    phase, Dphase, D2phase, D3phase, D4phase, D5phase = phase_derivs[0:6]
    i = 1j

    # Term_A = 5 * Damp * X
    X = (-1296 * i * freq**4 * D4phase
        - 3888 * freq**4 * D2phase**2
        + 1296 * freq**4 * Dphase**4
        + 6048 * i * freq**3 * D3phase
        - 6048 * i * freq**3 * Dphase**3
        - 19656 * i * freq**2 * D2phase
        + 216 * i * freq**2 * Dphase**2 * (36 * freq**2 * D2phase + 91 * i)
        - 24 * freq * Dphase * (216 * freq**3 * D3phase - 756 * freq**2 * D2phase - 1729 * i)
        + 43225)
    Term_A = 5 * Damp * X

    # Term_B = 12 * freq * [5 * D2amp * Y - 6 * freq * (5 * D3amp * Z + 3 * freq * F)]
    Y = (-216 * i * freq**3 * D3phase
        + 216 * i * freq**3 * Dphase**3
        + 756 * i * freq**2 * D2phase
        + 756 * freq**2 * Dphase**2
        - 18 * freq * Dphase * (36 * freq**2 * D2phase + 91 * i)
        - 1729)
    Z = (36 * i * freq**2 * D2phase
        + 36 * freq**2 * Dphase**2
        - 84 * i * freq * Dphase
        - 91)
    F = -6 * freq * D5amp + 5 * D4amp * (7 + 6 * i * freq * Dphase)
    Term_B = 12 * freq * (5 * D2amp * Y - 6 * freq * (5 * D3amp * Z + 3 * freq * F))

    # Term_C = expression multiplied by amp
    Term_C = (
        -7776 * i * freq**5 * Dphase**5
        + 136080 * freq**4 * D2phase**2
        - 45360 * freq**4 * Dphase**4
        - 360 * freq**2 * (216 * freq**3 * D3phase - 1729 * i) * D2phase
        + 2160 * freq**3 * Dphase**3 * (36 * freq**2 * D2phase + 91 * i)
        + 360 * freq**2 * Dphase**2 * (216 * i * freq**3 * D3phase -756 * i * freq**2 * D2phase + 1729)
        - i * (7776 * freq**5 * D5phase - 45360 * freq**4 * D4phase + 196560 * freq**3 * D3phase - 1339975 * i)
        + 30 * freq * Dphase * (-1296 * freq**4 * D4phase + 3888 * i * freq**4 * D2phase**2 + 6048 * freq**3 * D3phase -19656 * freq**2 * D2phase -43225 * i)
    )

    numerator = (6 * freq * (Term_A + Term_B) + amp * Term_C)
    denominator = 7776 * freq**(37/6)

    amp_5order = -numerator / denominator
    return amp_5order * np.exp(-1j * phase)

def waveform_deriv6_vector(amp_derivs, phase_derivs, f):
    """Return the sixth frequency derivative of the waveform."""
    amp, Damp, D2amp, D3amp, D4amp, D5amp, D6amp = amp_derivs
    phase, Dphase, D2phase, D3phase, D4phase, D5phase, D6phase = phase_derivs

    i = 1j
    # Polynomial subexpressions in the product-rule expansion.
    E1 = (7776 * Dphase**5 * f**5
        + 7776 * D5phase * f**5
        - 45360 * i * Dphase**4 * f**4
        + 136080 * i * D2phase**2 * f**4
        - 45360 * D4phase * f**4
        + 2160 * i * Dphase**3 * (36 * f**2 * D2phase + 91 * i) * f**3
        + 196560 * D3phase * f**3
        - 360 * Dphase**2 * (216 * f**3 * D3phase - 756 * f**2 * D2phase - 1729 * i) * f**2
        + 360 * D2phase * (-216 * i * f**3 * D3phase - 1729) * f**2
        - 30 * Dphase * (3888 * D2phase**2 * f**4 + 1296 * i * D4phase * f**4 - 6048 * i * D3phase * f**3 + 19656 * i * D2phase * f**2 - 43225)
        - 1339975 * i)

    E2 = (1296 * i * Dphase**4 * f**4
        - 3888 * i * D2phase**2 * f**4
        + 1296 * D4phase * f**4
        + 6048 * Dphase**3 * f**3
        - 6048 * D3phase * f**3
        + 19656 * D2phase * f**2
        - 216 * Dphase**2 * (36 * f**2 * D2phase + 91 * i) * f**2
        - 24 * i * Dphase * (216 * f**3 * D3phase - 756 * f**2 * D2phase - 1729 * i) * f
        + 43225 * i)

    E3 = (216 * Dphase**3 * f**3
        - 216 * D3phase * f**3
        - 756 * i * Dphase**2 * f**2
        + 756 * D2phase * f**2
        + 18 * i * Dphase * (36 * f**2 * D2phase + 91 * i) * f
        + 1729 * i)

    F = (5 * (36 * Dphase**2 * f**2 + 36 * i * D2phase * f**2 - 84 * i * Dphase * f - 91) * D4amp
        + 12 * f * ((6 * f * i * Dphase + 7) * D5amp - f * D6amp))

    T1 = 36 * f * i * (Damp * E1 + 3 * f * (5 * D2amp * E2 - 4 * f * (10 * D3amp * E3 + 9 * f * i * F)))

    E5 = (46656 * Dphase**6 * f**6
        - 699840 * i * D2phase**3 * f**6
        + 466560 * D3phase**2 * f**6
        + 46656 * i * D6phase * f**6
        - 326592 * i * Dphase**5 * f**5
        - 326592 * i * D5phase * f**5
        + 5307120 * D2phase**2 * f**4
        + 19440 * i * Dphase**4 * (36 * f**2 * D2phase + 91 * i) * f**4
        + 1769040 * i * D4phase * f**4
        - 7469280 * i * D3phase * f**3
        - 4320 * Dphase**3 * (216 * f**3 * D3phase - 756 * f**2 * D2phase - 1729 * i) * f**3
        + 540 * D2phase * (1296 * f**4 * D4phase - 6048 * f**3 * D3phase + 43225 * i) * f**2
        - 540 * Dphase**2 * (3888 * D2phase**2 * f**4 + 1296 * i * D4phase * f**4 - 6048 * i * D3phase * f**3 + 19656 * i * D2phase * f**2 - 43225) * f**2
        + 36 * Dphase * (7776 * f**5 * D5phase + 136080 * i * D2phase**2 * f**4 - 45360 * f**4 * D4phase + 196560 * f**3 * D3phase
            + 360 * D2phase * (-216 * i * f**3 * D3phase - 1729) * f**2 - 49579075) * f
        - 49579075)

    T2 = amp * E5

    amp_3order = - (T1 + T2) / (46656 * f**(43/6))
    return amp_3order * np.exp(-1j * phase)

def amp_deriv_vector(Mf: np.ndarray,
                     p: IMRPhenomDAmplitudeCoefficients,
                     prefactors: AmpInsPrefactors,
                     order: int) -> np.ndarray:
    """
    Return amplitude derivatives from order zero through ``order``.

    Each frequency region uses its corresponding inspiral, intermediate, or merger-ringdown ansatz before the arrays are concatenated.
    """
    # Set the amplitude-region join frequencies.
    p.fInsJoin = AMP_fJoin_INS
    p.fMRDJoin = p.fmaxCalc
    if p.fMRDJoin < p.fInsJoin:
        raise ValueError("fMRDJoin must be greater than fInsJoin")

    # The f**(-7/6) factor is differentiated separately.
    AmpPreFac = prefactors.amp0

    freqins, freqint, freqmr = split_freqs_amp(Mf, p)
    if len(freqins) > 0:
        powers_of_freqins = init_useful_power_arrays(freqins)
    if order >= 0:
        AmpIns, AmpInt, AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            AmpIns = AmpInsAnsatz_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            AmpInt = AmpIntAnsatz_vectorized(freqint, p)
        if len(freqmr) > 0:
            AmpMRD = AmpMRDAnsatz_vectorized(freqmr, p)
        # Concatenate the three frequency regions.
        Amp = np.concatenate((AmpIns, AmpInt, AmpMRD))
    if order >= 1:
        DAmpIns, DAmpInt, DAmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            DAmpIns = DAmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            DAmpInt = DAmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            DAmpMRD = DAmpMRD_vectorized(freqmr, p)
        # Concatenate the first derivatives from all frequency regions.
        DAmp = np.concatenate((DAmpIns, DAmpInt, DAmpMRD))
    if order >= 2:
        D2AmpIns, D2AmpInt, D2AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            D2AmpIns = D2AmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            D2AmpInt = D2AmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            D2AmpMRD = D2AmpMRD_vectorized(freqmr, p)
        # Concatenate the second derivatives from all frequency regions.
        D2Amp = np.concatenate((D2AmpIns, D2AmpInt, D2AmpMRD))
    if order >= 3:
        D3AmpIns, D3AmpInt, D3AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            D3AmpIns = D3AmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            D3AmpInt = D3AmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            D3AmpMRD = D3AmpMRD_vectorized(freqmr, p)
        # Concatenate the third derivatives from all frequency regions.
        D3Amp = np.concatenate((D3AmpIns, D3AmpInt, D3AmpMRD))
    if order >= 4:
        D4AmpIns, D4AmpInt, D4AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            D4AmpIns = D4AmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            D4AmpInt = D4AmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            D4AmpMRD = D4AmpMRD_vectorized(freqmr, p)
        # Concatenate the fourth derivatives from all frequency regions.
        D4Amp = np.concatenate((D4AmpIns, D4AmpInt, D4AmpMRD))
    if order >= 5:
        D5AmpIns, D5AmpInt, D5AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            D5AmpIns = D5AmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            D5AmpInt = D5AmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            D5AmpMRD = D5AmpMRD_vectorized(freqmr, p)
        # Concatenate the fifth derivatives from all frequency regions.
        D5Amp = np.concatenate((D5AmpIns, D5AmpInt, D5AmpMRD))
    if order >= 6:
        D6AmpIns, D6AmpInt, D6AmpMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmr)
        if len(freqins) > 0:
            D6AmpIns = D6AmpIns_vectorized(freqins, powers_of_freqins, prefactors)
        if len(freqint) > 0:
            D6AmpInt = D6AmpInt_vectorized(freqint, p)
        if len(freqmr) > 0:
            D6AmpMRD = D6AmpMRD_vectorized(freqmr, p)
        # Concatenate the sixth derivatives from all frequency regions.
        D6Amp = np.concatenate((D6AmpIns, D6AmpInt, D6AmpMRD))
    # Return all derivatives through the requested order.
    if order == 0:
        return AmpPreFac * np.array([Amp])
    elif order == 1:
        return AmpPreFac * np.array([Amp, DAmp])
    elif order == 2:
        return AmpPreFac * np.array([Amp, DAmp, D2Amp])
    elif order == 3:
        return AmpPreFac * np.array([Amp, DAmp, D2Amp, D3Amp])
    elif order == 4:
        return AmpPreFac * np.array([Amp, DAmp, D2Amp, D3Amp, D4Amp])
    elif order == 5:
        return AmpPreFac * np.array([Amp, DAmp, D2Amp, D3Amp, D4Amp, D5Amp])
    elif order == 6:
        return AmpPreFac * np.array([Amp, DAmp, D2Amp, D3Amp, D4Amp, D5Amp, D6Amp])
    else:
        raise ValueError("order must be between 0 and 6")

def phase_deriv_vector(Mf: np.ndarray,
                       p: IMRPhenomDPhaseCoefficients,
                       pn: PNPhasingSeries,
                       prefactors: PhiInsPrefactors,
                       Rholm: float,
                       Taulm: float,
                       order: int) -> np.ndarray:
    freqins, freqint, freqmrd = split_freqs_phase(Mf, p)
    if len(freqins) > 0:
        powers_of_freqins = init_useful_power_arrays(freqins)
    if order >= 0:
        PhiIns, PhiInt, PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            PhiIns = PhiInsAnsatzInt_vectorized(freqins, powers_of_freqins, prefactors, p, pn)
        if len(freqint) > 0:
            PhiInt = p.etaInv * PhiIntAnsatz_vectorized(freqint, p) + p.C1Int + p.C2Int * freqint
        if len(freqmrd) > 0:
            PhiMRD = p.etaInv * PhiMRDAnsatzInt_vectorized(freqmrd, p, Rholm, Taulm) + p.C1MRD + p.C2MRD * freqmrd
        Phi = np.concatenate((PhiIns, PhiInt, PhiMRD))
    if order >= 1:
        DPhiIns, DPhiInt, DPhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            DPhiIns = DPhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            DPhiInt = DPhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            DPhiMRD = DPhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the first derivatives from all frequency regions.
        DPhi = np.concatenate((DPhiIns, DPhiInt, DPhiMRD))
    if order >= 2:
        D2PhiIns, D2PhiInt, D2PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            D2PhiIns = D2PhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            D2PhiInt = D2PhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            D2PhiMRD = D2PhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the second derivatives from all frequency regions.
        D2Phi = np.concatenate((D2PhiIns, D2PhiInt, D2PhiMRD))
    if order >= 3:
        D3PhiIns, D3PhiInt, D3PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            D3PhiIns = D3PhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            D3PhiInt = D3PhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            D3PhiMRD = D3PhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the third derivatives from all frequency regions.
        D3Phi = np.concatenate((D3PhiIns, D3PhiInt, D3PhiMRD))
    if order >= 4:
        D4PhiIns, D4PhiInt, D4PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            D4PhiIns = D4PhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            D4PhiInt = D4PhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            D4PhiMRD = D4PhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the fourth derivatives from all frequency regions.
        D4Phi = np.concatenate((D4PhiIns, D4PhiInt, D4PhiMRD))
    if order >= 5:
        D5PhiIns, D5PhiInt, D5PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            D5PhiIns = D5PhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            D5PhiInt = D5PhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            D5PhiMRD = D5PhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the fifth derivatives from all frequency regions.
        D5Phi = np.concatenate((D5PhiIns, D5PhiInt, D5PhiMRD))
    if order >= 6:
        D6PhiIns, D6PhiInt, D6PhiMRD = np.zeros_like(freqins), np.zeros_like(freqint), np.zeros_like(freqmrd)
        if len(freqins) > 0:
            D6PhiIns = D6PhiIns_vectorized(freqins, powers_of_freqins, p, prefactors)
        if len(freqint) > 0:
            D6PhiInt = D6PhiInt_vectorized(freqint, p)
        if len(freqmrd) > 0:
            D6PhiMRD = D6PhiMRD_vectorized(freqmrd, p, Rholm, Taulm)
        # Concatenate the sixth derivatives from all frequency regions.
        D6Phi = np.concatenate((D6PhiIns, D6PhiInt, D6PhiMRD))
    # Return all derivatives through the requested order.
    if order == 0:
        return [Phi]
    elif order == 1:
        return [Phi, DPhi]
    elif order == 2:
        return [Phi, DPhi, D2Phi]
    elif order == 3:
        return [Phi, DPhi, D2Phi, D3Phi]
    elif order == 4:
        return [Phi, DPhi, D2Phi, D3Phi, D4Phi]
    elif order == 5:
        return [Phi, DPhi, D2Phi, D3Phi, D4Phi, D5Phi]
    elif order == 6:
        return [Phi, DPhi, D2Phi, D3Phi, D4Phi, D5Phi, D6Phi]
    else:
        raise ValueError("order must be between 0 and 6")


def IMRPhenomDAcc(freqs_in: np.ndarray, deltaF: float, phi0: float, fRef: float,
                          m1: float, m2: float, chi1_in: float, chi2_in: float,
                          distance: float, acc: float, tt0: float, order: int, extraParams: dict, NRTidal_version: str) -> np.ndarray:
    """
    Generate an accelerated IMRPhenomD frequency-domain waveform.

    A positive ``deltaF`` selects a uniform frequency grid. Component masses are in solar masses, distance is in metres, ``acc`` is the effective line-of-sight acceleration in s^-1, and ``order`` selects FSD order zero through three.
    """
    # Validate physical parameters and the requested frequency grid.
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

    # A zero reference frequency means the lower frequency bound.
    fRef_use = f_min if fRef == 0.0 else fRef

    # Total mass and symmetric mass ratio.
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    if eta > 0.25:
        eta = 0.25  # Guard against round-off above the equal-mass limit.
    if eta < 0.0 or eta > 0.25:
        raise ValueError("Unphysical eta; must be between 0 and 0.25.")

    # Convert the total mass to seconds for dimensionless Mf.
    M_sec = M * LAL_MTSUN_SI

    # Overall waveform-amplitude prefactor.
    amp0 = 2.0 * math.sqrt(5.0 / (64.0 * LAL_PI)) * M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance

    # Build the working frequency array and output offset.
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

    # Frequencies outside the requested interval remain zero.
    waveform = np.zeros(npts, dtype=complex)


    # Construct the IMRPhenomD model coefficients.
    finspin = final_spin0815(eta, chi1_in, chi2_in)
    if finspin < MIN_FINAL_SPIN:
        print(f"Warning: final spin ({finspin}) is small; model may misbehave.")

    pAmp = IMRPhenomDAmplitudeCoefficients()
    ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1_in, chi2_in, finspin)

    if extraParams is None:
        extraParams = {}
    # Match the aligned-spin TaylorF2 phasing convention.
    extraParams["PNSpinOrder"] = "35PN"

    pPhi = IMRPhenomDPhaseCoefficients()
    ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1_in, chi2_in, finspin, extraParams)

    # Generate the post-Newtonian phase series.
    pn = XLALSimInspiralTaylorF2AlignedPhasing(m1, m2, chi1_in, chi2_in, extraParams)
    if pn is None:
        raise RuntimeError("Failed to compute PN phasing series.")

    phi_prefactors = PhiInsPrefactors()
    status = init_phi_ins_prefactors(phi_prefactors, pPhi, pn)
    if status != 0:
        raise RuntimeError("init_phi_ins_prefactors failed.")

    ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, phi_prefactors, 1.0, 1.0)
    # Shift the waveform so that the peak amplitude is approximately at t=0.
    t0 = DPhiMRD(pAmp.fmaxCalc, pPhi, 1.0, 1.0)

    amp_prefactors = AmpInsPrefactors()
    status = init_amp_ins_prefactors(amp_prefactors, pAmp)
    if status != 0:
        raise RuntimeError("init_amp_ins_prefactors failed.")

    MfRef = M_sec * fRef_use
    powers_of_fRef = init_useful_powers(MfRef)
    phifRef = IMRPhenDPhase(MfRef, pPhi, pn, powers_of_fRef, phi_prefactors, 1.0, 1.0)
    phi_precalc = 2.0 * phi0 + phifRef

    # Evaluate the required derivatives on the frequency array.
    Mf = M_sec * freqs_used
    # FSD order n requires waveform derivatives through order 2n.
    deriv_order = 2*order
    amp_derivs = amp_deriv_vector(Mf, pAmp, amp_prefactors, deriv_order) * M_sec ** (-7/6)
    phase_derivs = phase_deriv_vector(Mf, pPhi, pn, phi_prefactors, 1.0, 1.0, deriv_order)
    # Convert derivatives with respect to Mf into frequency derivatives.
    amp_derivs = multiply_by_powers(amp_derivs, M_sec)
    phase_derivs = multiply_by_powers(phase_derivs, M_sec)
    phase_derivs[0] = phase_derivs[0] - t0 * (Mf - MfRef) - phi_precalc
    if order >= 1:
        phase_derivs[1] = phase_derivs[1] - t0 * M_sec
    # Assemble the accelerated waveform order by order.
    if order >= 0:
        wav = waveform_deriv0_vector(amp_derivs, phase_derivs, freqs_used)
        acc_wav = wav
    if order >= 1:
        deriv1 = waveform_deriv1_vector(amp_derivs, phase_derivs, freqs_used)
        deriv2 = waveform_deriv2_vector(amp_derivs, phase_derivs, freqs_used)
        acc_wav = acc_wav + acc / (2*np.pi) * (-1j*freqs_used*deriv2 + (4*np.pi*freqs_used*tt0 - 2j)*deriv1 + 4*np.pi*tt0*acc_wav*(1 + 1j*np.pi*freqs_used*tt0))
    if order >= 2:
        deriv3 = waveform_deriv3_vector(amp_derivs, phase_derivs, freqs_used)
        deriv4 = waveform_deriv4_vector(amp_derivs, phase_derivs, freqs_used)
        acc_wav = acc_wav - acc**2 * (
            freqs_used**2 * deriv4
            + 8j*np.pi*freqs_used**2*tt0*deriv3
            - 24*np.pi**2*freqs_used**2*tt0**2*deriv2
            - 16j*np.pi*tt0*(2*np.pi**2*freqs_used**2*tt0**2 - 6j*np.pi*freqs_used*tt0 - 3)*deriv1
            + 16*np.pi**2*tt0**2*acc_wav*(np.pi**2*freqs_used**2*tt0**2 - 4j*np.pi*freqs_used*tt0 - 3)
            + 8*freqs_used*deriv3
            + 48j*np.pi*freqs_used*tt0*deriv2
            + 12*deriv2
        ) / (8*np.pi**2)
    if order == 3:
        deriv5 = waveform_deriv5_vector(amp_derivs, phase_derivs, freqs_used)
        deriv6 = waveform_deriv6_vector(amp_derivs, phase_derivs, freqs_used)
        acc_wav =  acc_wav + acc**3 * (
            -96*np.pi**2*tt0**2*(2*np.pi**3*freqs_used**3*tt0**3 - 15j*np.pi**2*freqs_used**2*tt0**2 - 30*np.pi*freqs_used*tt0 + 15j)*deriv1
            + 1j*(freqs_used*(freqs_used*(freqs_used*deriv6 + 6*deriv5*(3 + 2j*np.pi*freqs_used*tt0)) - 30*deriv4*(2*np.pi**2*freqs_used**2*tt0**2 - 6j*np.pi*freqs_used*tt0 - 3))
                + 40*deriv3*(-4j*np.pi**3*freqs_used**3*tt0**3 - 18*np.pi**2*freqs_used**2*tt0**2 + 18j*np.pi*freqs_used*tt0 + 3)
                + 240*np.pi*tt0*(np.pi**3*freqs_used**3*tt0**3 - 6j*np.pi**2*freqs_used**2*tt0**2 - 9*np.pi*freqs_used*tt0 + 3j)*deriv2)
            + 32*np.pi**3*tt0**3*acc_wav*(-2j*np.pi**3*freqs_used**3*tt0**3 - 18*np.pi**2*freqs_used**2*tt0**2 + 45j*np.pi*freqs_used*tt0 + 30)
        ) / (48*np.pi**3)
    if order > 3 or order < 0:
        raise ValueError("order must be between 0 and 3 for IMRPhenomDAcc.")
    waveform[offset:offset + len(freqs_used)] = amp0 * acc_wav

    return waveform

def XLALSimIMRPhenomDAccFrequencySequence(phi0: float, fRef_in: float, deltaF: float,
                                        m1: float, m2: float,
                                        chi1: float, chi2: float,
                                        distance_mpc: float, acc: float, t0: float, order: int,
                                        freqs: list[float],
                                        extraParams, NRTidal_version) -> np.ndarray:
    """
    Return the accelerated IMRPhenomD frequency-domain plus polarization.

    Component masses are in solar masses, distance is in Mpc, ``acc`` is in s^-1, ``order`` is zero through three, and ``freqs`` supplies [f_min, f_max] in Hz.
    """
    # Validate the public wrapper inputs.
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
    if not (0 <= order <= 3):
        raise ValueError("order must be between 0 and 3")

    # Convert distance from Mpc to metres.
    distance = distance_mpc * LAL_MPC_SI

    # Warn outside the calibrated mass-ratio range.
    q = m1 / m2 if m1 > m2 else m2 / m1
    if q > MAX_ALLOWED_MASS_RATIO:
        print("Warning: The model is not supported for high mass ratio (q > {}), but continue.".format(MAX_ALLOWED_MASS_RATIO))

    # A zero reference frequency means the lower frequency bound.
    f_min = freqs[0]
    f_max = freqs[1]
    fRef = f_min if fRef_in == 0.0 else fRef_in

    # Convert the total mass to seconds for dimensionless Mf.
    M_sec = (m1 + m2) * LAL_MTSUN_SI
    # Convert the dimensionless model cutoff to hertz.
    fCut = f_CUT / M_sec
    if fCut <= f_min:
        raise ValueError(f"(fCut = {fCut} Hz) <= f_min = {f_min} Hz")

    # A zero upper bound selects the model cutoff.
    f_max_prime = f_max if f_max != 0 else fCut
    f_max_prime = min(f_max_prime, fCut)
    if f_max_prime <= f_min:
        raise ValueError("f_max_prime <= f_min")

    # The core routine constructs the complete uniform grid.
    freqs_boundary = [f_min, f_max_prime]

    waveform = IMRPhenomDAcc(freqs_boundary, deltaF, phi0, fRef,
                                          m1, m2, chi1, chi2, distance,
                                          acc, t0, order,
                                          extraParams, NRTidal_version)

    # Pad to the requested upper frequency when it exceeds the model cutoff.
    if f_max_prime < f_max:
        n_full = next_pow2(f_max / deltaF) + 1
        waveform_full = np.zeros(n_full, dtype=complex)
        n_current = waveform.size
        waveform_full[:n_current] = waveform
        waveform = waveform_full

    return waveform

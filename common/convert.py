import numpy as np


def cover_bps_to_Mbps(bps: float) -> float:
    return bps / 1000000


def cover_Mbps_to_bps(Mbps: float) -> float:
    return Mbps * 1000000


def cover_MHz_to_Hz(MHz: float) -> float:
    return MHz * 1e6


def cover_ratio_to_dB(ratio: float) -> float:
    return 10 * np.log10(ratio)


def cover_dB_to_ratio(dB: float) -> float:
    return np.power(10, (dB / 10))


def cover_dBm_to_W(dBm: float) -> float:
    return np.power(10, (dBm / 10)) / 1000


def cover_W_to_dBm(W: float) -> float:
    return 10 * np.log10(W * 1000)


def cover_W_to_mW(W: float) -> float:
    return W * 1000


def cover_mW_to_W(mW: float) -> float:
    return mW / 1000


def cover_GHz_to_Hz(GHz: float) -> float:
    return GHz * 1e9


def cover_Hz_to_GHz(GHz: float) -> float:
    return GHz / 1e9


def cover_MB_to_b(MB: float) -> float:
    return MB * 1024 * 1024 * 8


def cover_b_to_MB(b: float) -> float:
    return b / (1024 * 1024 * 8)


def cover_b_to_KB(b: float) -> float:
    return b / (1024 * 8)

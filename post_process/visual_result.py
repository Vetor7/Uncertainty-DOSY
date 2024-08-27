from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt

def contourDOSY(data, Type, fig, axs, Z, spec_var, diff_mean_list, merged_boxes):
    scale_list = {
        "QGC": (4, 12.5, 1, 12),
        "GSP": (3, 5.5, 1.5, 5),
        "M6": (0.5, 5.5, 2, 12),
        "QDC": (0.5, 3, 1, 15),
        "VD": (1.8, 2.5, 12, 16),
        "TSP": (-1, 5, 3, 16),
        "TD": (-1, 2.5, 1, 14),
        "BPP1": (1, 5, 6, 13),
        "BPP2": (1, 5, 6, 13),
        "SMDn_high": (0, 9, 0, 1),
        "SMDn_low": (0, 9, 0, 14),
        "new": (-1, 8, -1, 21),
        "AQM": (0.5, 6.2, 4, 14),
        "QG": (1, 9, 1, 7),
        "QG2": (1, 9, 1, 7),
        "JNN": (-2.5, 2.6, 2, 14),
        "JNN109": (-2.5, 2.6, 2, 14),
        "JNN228": (-2.5, 2.6, 2, 14),
        "JNN330": (-2.5, 2.6, 2, 14),
        "EC": (1, 10, 1, 5),
        "AMDK": (1, 10, 1, 5)
    }

    if Type not in scale_list:
        raise ValueError(f"Type {Type} is not recognized.")
    
    cs1, cs2, dc1, dc2 = scale_list[Type]

    ppm = data['ppm']
    idx_peaks = data['idx_peaks']
    HNMR = data['HNMR']
    
    b = data['b']
    cs_spec = np.zeros([(ppm.size), 1])
    spec_whole = np.zeros([len(Z[0, :]), ppm.size])
    cs_spec[idx_peaks, :] = HNMR
    spec_whole[0:140, idx_peaks[0, :]] = Z.T
    decay_range = np.linspace(0, (len(Z[0, :]) - 1) / 10, len(Z[0, :]))
    axs[0].plot(ppm.reshape(ppm.size), cs_spec, color='k')
    axs[0].set_xlim([cs1, cs2])
    axs[0].invert_xaxis()
    axs[0].axis('off')
                               
    scale_factor = 0.8 if Type != "VD" else 0.12
    CS, DC = np.meshgrid(ppm.reshape(ppm.size), decay_range * (scale_factor / b[0, -1]))
    axs[1].contour(CS, DC, spec_whole, levels=40)
    axs[1].set_xlim([cs1, cs2])
    axs[1].set_ylim([dc1, dc2])
    axs[1].invert_yaxis()
    axs[1].invert_xaxis()

    y_indices = np.arange(len(decay_range))
    for bb, diff in zip(merged_boxes, diff_mean_list):
        if diff == 0:
            continue
        (x1, y1), (x2, y2) = bb

        try:
            x1_mapped = ppm[0,x1]
            x2_mapped = ppm[0,x2]
            y1_mapped = np.interp(y1, y_indices, decay_range* (0.8 / b[0, -1]))
            y2_mapped = np.interp(y2, y_indices, decay_range* (0.8 / b[0, -1]))
        except Exception as e:
            print("Error during interpolation:", e)
            continue

        width = x2_mapped - x1_mapped
        height = y2_mapped - y1_mapped
        centre_x = (x2_mapped + x1_mapped)/2
        horinze_d = (dc2-dc1)

        color, symbols = ('green', '✓') if diff <= 0.5 else ('red', '?')
        fontsizes = 14

        rect = Rectangle((x1_mapped, y1_mapped), width, height, linewidth=1, edgecolor=color, facecolor='none')
        axs[1].add_patch(rect)
        axs[1].annotate(symbols, (centre_x, y1_mapped-0.04*horinze_d), color=color, fontsize=fontsizes, ha='center', va='center')

    plt.xlabel('Chemical Shift(ppm)')
    fig.text(0.04, 0.4, 'Diffusion Coefficient(10^(-10) m^2/s)', va='center', rotation='vertical')

    pass

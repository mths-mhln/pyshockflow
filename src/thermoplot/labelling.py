###########################################
# Imports
###########################################
import numpy as np





###########################################
# Labelling
###########################################
def compute_label_coord_and_angle(ax, isoline_data, frac, y_scale):
    # extract coordinates from isoline data for clarity of the code. 
    x = isoline_data["coords"][:, 0]
    y = isoline_data["coords"][:, 1]

    # if isoline is too short, skip labelling as it will likely not properly show.
    if len(x) < 4:
        return None
    
    # Get index of the isoline coordinate at which the label is desired to be placed. 
    idx = int(np.clip(frac * len(x), 1, len(x)-2))

    # store label coord in np array
    label_coord = np.array([x[idx], y[idx]])

    # extract plot boundaries and normalize coordinates to [0,1] for angle calculation to be independent of axis limits and units.
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    x_norm = (x - xl) / (xh - xl)
    if y_scale == 'log': # PH diagram has log scale for the pressure
        log_yl, log_yh = np.log10(yl), np.log10(yh)
        y_norm = (np.log10(np.abs(y) + 1e-300) - log_yl) / (log_yh - log_yl)
    else:
        y_norm = (y - yl) / (yh - yl)

    # For the angle to look visually correct in the image, the angle should be determined according to the physical size of the 
    # matplotlib window that appears for it to look aligned with the isoline. Note that it is not the figsize we need, but the
    # size in inches of the axes
    ax_spine_w, ax_spine_h = ax.get_window_extent().width, ax.get_window_extent().height

    # Get angle in physical space
    dx = (x_norm[idx+1] - x_norm[idx-1]) * ax_spine_w
    dy = (y_norm[idx+1] - y_norm[idx-1]) * ax_spine_h
    label_ang = np.degrees(np.arctan2(dy, dx))

    # Normalise to [-90, 90] so text is always right-reading. if angle exceeds this region, ensure correct orientation through the flip
    flip = False
    if (label_ang > 90 and label_ang < 180) or (label_ang < -90 and label_ang > -180):
        flip = True
    label_ang = ((label_ang + 90) % 180) - 90
    if flip:
        label_ang += 180
    return label_coord, label_ang

    

def draw_isolines_labeled(ax, isolines_data, color, frac, lbl_fmt_short, lbl_fmt_named, yscale='linear'):
    # if no data is passed to the function, draw nothing.
    if not isolines_data:
        return

    # compute isoline coordinate at which to place label according to specified fraction value, and
    # compute isoline label angle at said coordinate such that the label orientation matches the local slope.
    # this will become a list of tuples (label_coord, label_ang) for each isoline.
    labels_data = [compute_label_coord_and_angle(ax, isoline_data, frac, yscale) for isoline_data in isolines_data]

    # make central isoline the named one: extract index of this label
    idx_named_label = len(isolines_data) // 2

    # plot isolines with appropriate labels.
    for k, ((isoline_data, label_data)) in enumerate(zip(isolines_data, labels_data)):
        iv_arr, dv_arr = isoline_data["coords"][:, 0], isoline_data["coords"][:, 1]
        ax.plot(iv_arr, dv_arr, color=color, lw=0.6, zorder=2)
        if label_data is None:
            continue
        label_coord, label_ang = label_data
        lbl = lbl_fmt_named(isoline_data["isoline_val"]) if k == idx_named_label else lbl_fmt_short(isoline_data["isoline_val"])

        # if isoline is long enough for label to be visible, put label at that coordinate with appropriate angle.
        if len(iv_arr) >= 4:
            ax.text(label_coord[0], label_coord[1], lbl,
            color=color, fontsize=10, rotation=label_ang, rotation_mode='anchor',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6, clip_on=True)
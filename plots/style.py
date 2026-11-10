# Shared plot style: every algorithm keeps the same color and line style in every figure
# Each algorithm has a fixed color

ALGORITHM_STYLE = {
    "llamea_bbob":   {"color": "#2a78d6", "linestyle": "-",  "label": "LLaMEA (evolved on f15)"},
    "llamea_real":   {"color": "#eb6834", "linestyle": "-",  "label": "LLaMEA (evolved on 1124)"},
    "botorch_bo":    {"color": "#1baf7a", "linestyle": "--", "label": "BoTorch BO"},
    "cmaes":         {"color": "#6250d6", "linestyle": "-.", "label": "CMA-ES"},
    "random_search": {"color": "#8a8a85", "linestyle": ":",  "label": "Random search"},
}

PROBLEM_TITLE = {
    "bbob": "BBOB f15 (Rastrigin), d = 6",
    "real": "1124 FM Sound Waves, d = 6",
}


def style_for(name):
    # Style of an algorithm. An algorithm that is not in the list gets pink and its folder name.
    return ALGORITHM_STYLE.get(name, {"color": "#e87ba4", "linestyle": "-", "label": name})


def clean_axes(ax):
    # Light grid, no top and right border.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color="#e5e5e0", linewidth=0.8)
    ax.set_axisbelow(True)

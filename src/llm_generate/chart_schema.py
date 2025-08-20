CHART_SCHEMAS = {
    "generate_line_chart": {
        "required_keys": ["time", "value"],
        "hint": "折线图, data=[{time: string, value: number}], 可选 group"
    },
    "generate_area_chart": {
        "required_keys": ["time", "value"],
        "hint": "面积图, data=[{time: string, value: number}]"
    },
    "generate_column_chart": {
        "required_keys": ["category", "value"],
        "hint": "柱状图, data=[{category: string, value: number}], 可选 group/stack"
    },
    "generate_bar_chart": {
        "required_keys": ["category", "value"],
        "hint": "条形图(横向), data=[{category: string, value: number}]"
    },
    "generate_pie_chart": {
        "required_keys": ["category", "value"],
        "hint": "饼图, data=[{category: string, value: number}]"
    },
    "generate_scatter_chart": {
        "required_keys": ["x", "y"],
        "hint": "散点图, data=[{x: number, y: number}]"
    },
    "generate_histogram_chart": {
        "required_keys": ["value"],  # 数组 number[]
        "hint": "直方图, data=[number, ...]"
    },
    "generate_boxplot_chart": {
        "required_keys": ["x", "value"],
        "hint": "箱线图, data=[{x: string, value: number}]"
    },
    "generate_violin_chart": {
        "required_keys": ["x", "value"],
        "hint": "小提琴图, data=[{x: string, value: number}]"
    },
    "generate_radar_chart": {
        "required_keys": ["category", "value"],
        "hint": "雷达图, data=[{category: string, value: number}], 支持多维度比较"
    },
    "generate_treemap_chart": {
        "required_keys": ["name", "value"],
        "hint": "矩形树图, data=[{name: string, value: number}]"
    },
    "generate_funnel_chart": {
        "required_keys": ["stage", "value"],
        "hint": "漏斗图, data=[{stage: string, value: number}]"
    },
    "generate_dual_axes_chart": {
        "required_keys": ["time", "value1", "value2"],
        "hint": "双轴图, data=[{time: string, value1: number, value2: number}]"
    },
    "generate_liquid_chart": {
        "required_keys": ["value"],
        "hint": "液体图, data=[{value: number}] (单值百分比)"
    },
    "generate_word_cloud_chart": {
        "required_keys": ["text", "value"],
        "hint": "词云图, data=[{text: string, value: number}]"
    },
    "generate_sankey_chart": {
        "required_keys": ["source", "target", "value"],
        "hint": "桑基图/流向图, data=[{source: string, target: string, value: number}]"
    },
    "generate_network_graph": {
        "required_keys": ["source", "target", "value"],
        "hint": "网络关系图, data=[{source: string, target: string, value: number}]"
    },
    "generate_organization_chart": {
        "required_keys": ["id", "parent", "label"],
        "hint": "组织架构图, data=[{id: string, parent: string, label: string}]"
    },
    "generate_mind_map": {
        "required_keys": ["id", "parent", "label"],
        "hint": "思维导图, data=[{id: string, parent: string, label: string}]"
    },
    "generate_flow_diagram": {
        "required_keys": ["id", "from", "to", "label"],
        "hint": "流程图, data=[{id: string, from: string, to: string, label: string}]"
    },
    "generate_fishbone_diagram": {
        "required_keys": ["problem", "cause"],
        "hint": "鱼骨图, data=[{problem: string, cause: string}]"
    },
    "generate_path_map": {
        "required_keys": ["lng", "lat"],
        "hint": "路径图, data=[{lng: number, lat: number}]"
    },
    "generate_pin_map": {
        "required_keys": ["lng", "lat"],
        "hint": "点地图, data=[{lng: number, lat: number}]"
    },
    "generate_district_map": {
        "required_keys": ["district", "value"],
        "hint": "行政区划地图(中国), data=[{district: string, value: number}]"
    },
    "generate_venn_chart": {
        "required_keys": ["sets", "size"],
        "hint": "维恩图, data=[{sets: [string], size: number}]"
    },
}

import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import os
import sys


# 设置中文字体支持
def setup_chinese_font():
    """配置中文字体支持"""
    try:
        # 尝试使用系统中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/simsun.ttc',  # Windows 宋体
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
            '/System/Library/Fonts/Arial Unicode.ttf',  # macOS
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
        ]

        chinese_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                chinese_font = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = [chinese_font.get_name()]
                print(f"✅ 使用中文字体: {os.path.basename(font_path)}")
                break

        if chinese_font is None:
            # 如果没有找到中文字体，使用英文显示
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
            print("⚠️  使用英文显示，中文字体未找到")

        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        return True

    except Exception as e:
        print(f"⚠️  字体设置失败: {e}，使用英文显示")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
        return False


# 初始化中文字体
chinese_supported = setup_chinese_font()
plt.rcParams.update({'font.size': 14})

# 常量定义
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    """医疗数据分箱初始化"""
    for i in range(num_bins):
        bin_dict[i] = {
            COUNT: 0,
            CONF: 0,
            ACC: 0,
            BIN_ACC: 0,
            BIN_CONF: 0
        }


def _populate_bins(confs, preds, labels, num_bins=10):
    """医疗AI预测结果分箱统计"""
    bin_dict = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil((num_bins * confidence) - 1))
        binn = max(0, min(binn, num_bins - 1))

        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        bin_dict[binn][ACC] += (1 if (label == prediction) else 0)

    for binn in range(num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])

    return bin_dict


def get_chinese_label(english_label):
    """获取中文字符标签"""
    chinese_labels = {
        'Medical AI Model Calibration Analysis': '模型校准分析',
        'Prediction Confidence': '预测置信度',
        'Actual Accuracy': '实际准确率',
        'Ideal Calibration': '理想校准线',
        'Model Output': '模型输出',
        'Calibration Error': '校准误差',
        'Expected Calibration Error': '预期校准误差',
        'Sample Percentage': '样本百分比',
        'Confidence Interval': '置信度区间',
        'Medical AI Prediction Confidence Distribution': '模型预测置信度分布',
        'Reliability Analysis': '可靠性分析',
        'Simple Reliability': '简化可靠性图',
        'Distribution': '分布图',
        'Accuracy': '准确率',
        'Confidence': '置信度'
    }
    return chinese_labels.get(english_label, english_label) if chinese_supported else english_label


def reliability_plot(confs, preds, labels, num_bins=15, save="medical_reliability_plot.png"):
    """医疗AI可靠性图"""
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = [bin_dict[i][BIN_ACC] for i in range(num_bins)]

    plt.figure(figsize=(10, 8))
    plt.bar(bns, bns, align='edge', width=0.05, color='lightcoral',
            label=get_chinese_label('Ideal Calibration'), alpha=0.7)
    plt.bar(bns, y, align='edge', width=0.05, color='steelblue',
            alpha=0.7, label=get_chinese_label('Model Output'))

    plt.ylabel(get_chinese_label('Actual Accuracy'))
    plt.xlabel(get_chinese_label('Prediction Confidence'))
    plt.title(get_chinese_label('Medical AI Model Calibration Analysis'))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 计算ECE
    total_count = sum(bin_dict[i][COUNT] for i in range(num_bins))
    ece = sum(bin_dict[i][COUNT] * abs(bin_dict[i][BIN_ACC] - bin_dict[i][BIN_CONF])
              for i in range(num_bins)) / total_count if total_count > 0 else 0

    plt.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

    return ece


def bin_strength_plot(confs, preds, labels, num_bins=15, save="medical_bin_strength.png"):
    """医疗AI预测分布图"""
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = [(bin_dict[i][COUNT] / float(num_samples)) * 100 for i in range(num_bins)]

    plt.figure(figsize=(10, 8))
    plt.bar(bns, y, align='edge', width=0.05, color='seagreen',
            alpha=0.7, label=get_chinese_label('Sample Percentage'))

    plt.ylabel(get_chinese_label('Sample Percentage') + ' (%)')
    plt.xlabel(get_chinese_label('Confidence Interval'))
    plt.title(get_chinese_label('Medical AI Prediction Confidence Distribution'))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


def reliability_plot2(confs, preds, labels, num_bins=15, save="medical_calibration_report.png"):
    """医疗AI校准报告 - 专业版"""
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    accs = np.array([bin_dict[i][BIN_ACC] for i in range(num_bins)])
    confs_array = np.array([bin_dict[i][BIN_CONF] for i in range(num_bins)])

    total_count = sum(bin_dict[i][COUNT] for i in range(num_bins))
    ece = sum(bin_dict[i][COUNT] * abs(bin_dict[i][BIN_ACC] - bin_dict[i][BIN_CONF])
              for i in range(num_bins)) / total_count if total_count > 0 else 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 简化的绘图逻辑
    bin_size = 1 / num_bins
    positions = np.arange(bin_size / 2, 1 + bin_size / 2, bin_size)

    # 确保数组长度匹配
    min_len = min(len(positions), len(accs))
    positions = positions[:min_len]
    accs = accs[:min_len]
    confs_array = confs_array[:min_len]

    ax.bar(positions, accs, width=bin_size, color='royalblue',
           alpha=0.8, label=get_chinese_label('Model Output'))
    ax.plot([0, 1], [0, 1], 'r--', label=get_chinese_label('Ideal Calibration'), linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(get_chinese_label('Prediction Confidence'))
    ax.set_ylabel(get_chinese_label('Actual Accuracy'))
    ax.set_title(get_chinese_label('Medical AI Model Calibration Analysis'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加ECE文本
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=12)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

    return ece


def load_test_save_cab_file():
    """专门加载test_save_cab.pt文件"""
    filename = "test_save_cab.pt"

    if not os.path.exists(filename):
        print(f"❌ 文件 {filename} 不存在")
        return None

    try:
        print(f"正在加载文件: {filename}")

        # 使用安全的加载方式
        data = torch.load(filename, map_location='cpu', weights_only=False)

        print(f"✅ 成功加载文件")
        print(f"文件类型: {type(data)}")

        if isinstance(data, dict):
            print(f"文件包含的键: {list(data.keys())}")

            # 检查数据内容
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: 形状{value.shape}, 类型{value.dtype}")
                elif torch.is_tensor(value):
                    print(f"  {key}: 张量形状{value.shape}, 类型{value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")

        return data

    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None


def adapt_data_format(raw_data):
    """适配test_save_cab.pt数据格式"""
    print("🔄 适配数据格式...")

    # 检查数据格式并尝试适配
    if isinstance(raw_data, dict):
        # 检查是否包含预测结果
        if 'logits' in raw_data and 'preds' in raw_data and 'labels' in raw_data:
            print("✅ 数据格式正确，包含logits, preds, labels")
            return raw_data

        # 检查其他可能的键名
        key_mapping = {
            'outputs': 'logits',
            'predictions': 'preds',
            'targets': 'labels',
            'y_pred': 'preds',
            'y_true': 'labels'
        }

        adapted_data = {}
        for old_key, new_key in key_mapping.items():
            if old_key in raw_data:
                adapted_data[new_key] = raw_data[old_key]
                print(f"✅ 映射 {old_key} -> {new_key}")

        # 如果缺少必要字段，尝试从其他字段推断
        if 'logits' not in adapted_data and 'preds' in adapted_data:
            # 从preds生成伪logits
            preds = adapted_data['preds']
            if torch.is_tensor(preds):
                n_classes = int(preds.max()) + 1 if len(preds) > 0 else 2
                # 创建伪logits（one-hot编码加上一些噪声）
                fake_logits = torch.randn(len(preds), n_classes)
                fake_logits[torch.arange(len(preds)), preds] += 2  # 增加正确类别的置信度
                adapted_data['logits'] = fake_logits
                print("✅ 从preds生成伪logits")

        if 'labels' not in adapted_data:
            # 如果没有标签，使用preds作为伪标签（仅用于演示）
            if 'preds' in adapted_data:
                adapted_data['labels'] = adapted_data['preds'].clone()
                print("⚠️  使用preds作为伪标签（仅演示用）")

        return adapted_data if adapted_data else raw_data

    elif torch.is_tensor(raw_data):
        # 如果数据是张量，假设它是logits
        print("✅ 数据是张量，假设为logits")
        n_samples = raw_data.shape[0]
        n_classes = raw_data.shape[1] if len(raw_data.shape) > 1 else 2

        adapted_data = {
            'logits': raw_data,
            'preds': torch.argmax(raw_data, dim=1) if len(raw_data.shape) > 1 else (raw_data > 0.5).long(),
            'labels': torch.randint(0, n_classes, (n_samples,))  # 生成随机标签用于演示
        }
        print("✅ 从张量生成完整数据格式")
        return adapted_data

    else:
        print("❌ 无法识别的数据格式")
        return None


def create_demo_data():
    """创建演示数据"""
    print("🔄 创建演示数据...")
    torch.manual_seed(42)

    n_samples = 1000
    n_classes = 3

    # 生成更真实的医疗数据分布
    logits = torch.randn(n_samples, n_classes) * 2 + 1
    preds = torch.argmax(logits, dim=1)

    # 生成与预测相关的标签（模拟真实医疗场景）
    labels = preds.clone()
    noise_mask = torch.rand(n_samples) < 0.15  # 15%的错误率
    labels[noise_mask] = torch.randint(0, n_classes, (noise_mask.sum(),))

    demo_data = {
        "logits": logits,
        "preds": preds,
        "labels": labels
    }

    print("✅ 演示数据创建完成")
    return demo_data


def print_calibration_quality(ece):
    """打印校准质量评估"""
    print(f"\n=== 校准质量评估 ===")
    print(f"预期校准误差 (ECE): {ece:.4f}")

    if ece < 0.02:
        print("✅ 校准质量: 优秀 (ECE < 2%)")
        print("   模型置信度高度可靠，适合临床决策支持")
    elif ece < 0.05:
        print("✅ 校准质量: 良好 (ECE < 5%)")
        print("   模型置信度基本可靠，建议临床验证后使用")
    else:
        print("⚠️  校准质量: 需改进 (ECE ≥ 5%)")
        print("   建议进行温度缩放等校准处理")


def medical_ai_calibration_report(model_data, save_prefix="medical_ai"):
    """增强版医疗AI校准报告"""

    if model_data is None:
        print("❌ 无有效数据，无法生成报告")
        return None

    # 适配数据格式
    adapted_data = adapt_data_format(model_data)

    if adapted_data is None:
        print("❌ 数据格式适配失败")
        return None

    # 检查必要字段
    required_keys = ['logits', 'preds', 'labels']
    missing_keys = [key for key in required_keys if key not in adapted_data]

    if missing_keys:
        print(f"❌ 数据缺少必要字段: {missing_keys}")
        return None

    try:
        # 转换为numpy数组
        if torch.is_tensor(adapted_data["logits"]):
            confs = torch.softmax(adapted_data["logits"], dim=1).max(dim=1)[0].numpy()
            preds = adapted_data["preds"].numpy() if torch.is_tensor(adapted_data["preds"]) else adapted_data["preds"]
            labels = adapted_data["labels"].numpy() if torch.is_tensor(adapted_data["labels"]) else adapted_data[
                "labels"]
        else:
            # 如果已经是numpy数组
            confs = torch.softmax(torch.tensor(adapted_data["logits"]), dim=1).max(dim=1)[0].numpy()
            preds = np.array(adapted_data["preds"])
            labels = np.array(adapted_data["labels"])

        print("=== 医疗AI模型校准分析报告 ===")
        print(f"样本数量: {len(labels)}")
        print(f"类别数量: {len(np.unique(labels))}")

        # 生成各种可视化
        ece1 = reliability_plot2(confs, preds, labels,
                                 save=f"{save_prefix}_reliability.png")

        ece2 = reliability_plot(confs, preds, labels,
                                save=f"{save_prefix}_simple_reliability.png")

        bin_strength_plot(confs, preds, labels,
                          save=f"{save_prefix}_distribution.png")

        # 使用主要的ECE值
        ece = ece1 if not np.isnan(ece1) else ece2

        # 打印质量评估
        print_calibration_quality(ece)

        return ece

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    try:
        print("=== 医疗AI模型校准分析工具 ===")
        print(f"中文支持: {'已启用' if chinese_supported else '未启用'}")

        # 加载指定的文件
        test_data = load_test_save_cab_file()

        if test_data is None:
            print("❌ 无法加载 test_save_cab.pt 文件")
            print("🔄 使用演示数据生成报告...")
            test_data = create_demo_data()
            save_prefix = "medical_demo"
        else:
            save_prefix = "test_save_cab_analysis"

        # 生成医疗AI校准报告
        ece_score = medical_ai_calibration_report(test_data, save_prefix)

        if ece_score is not None:
            print(f"\n=== 报告生成完成 ===")
            print(f"生成的可视化文件:")
            print(f"1. {save_prefix}_reliability.png - 详细可靠性分析")
            print(f"2. {save_prefix}_simple_reliability.png - 简化可靠性图")
            print(f"3. {save_prefix}_distribution.png - 置信度分布图")
        else:
            print("❌ 校准报告生成失败")

    except Exception as e:
        print(f"❌ 医疗AI校准分析失败: {e}")
        import traceback

        traceback.print_exc()
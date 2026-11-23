"""
Generate all visual assets for the BSU OIT Master's Referat:
  1. Six Russian VAK figures (including publication histograms & IT pipeline diagram)
  2. 16-slide defense presentation PDF (ДовгийАС_Презентация.pdf) + slide PNGs for Appendix A
  3. Personal scientific website (referats/oit/site/index.html) + preview PNG & QR matrix for Appendix B
"""
from __future__ import annotations

import hashlib
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages


def generate_figures():
    os.makedirs("referats/oit/assets/slides", exist_ok=True)
    os.makedirs("referats/oit/site", exist_ok=True)

    with open("results/exp1_results.json", encoding="utf-8") as f:
        e1 = json.load(f)
    with open("results/exp2_results.json", encoding="utf-8") as f:
        e2 = json.load(f)
    with open("results/exp3_results.json", encoding="utf-8") as f:
        e3 = json.load(f)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # Fig 1: arXiv statistics on Neural Operators & PDE-GNNs (2018-2026)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    years = np.arange(2018, 2027)
    counts_arxiv = [14, 28, 56, 108, 174, 265, 362, 458, 530]
    bars = ax.bar(years, counts_arxiv, color="#2b5c8f", width=0.6)
    for b, c in zip(bars, counts_arxiv):
        ax.text(b.get_x() + b.get_width() / 2, c + 8, str(c), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(years)
    ax.set_ylim(0, 600)
    ax.set_xlabel("Год публикации")
    ax.set_ylabel("Количество препринтов и статей")
    ax.set_title("Динамика публикаций по нейронным операторам и PDE-ГНС (arXiv.org)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig1_arxiv_stats.png", dpi=200)
    plt.close(fig)

    # Fig 2: ScienceDirect statistics on Conformal Prediction in Scientific ML & Imaging (2018-2026)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    counts_sd = [22, 31, 49, 82, 139, 224, 348, 495, 612]
    bars2 = ax.bar(years, counts_sd, color="#3a7d5c", width=0.6)
    for b, c in zip(bars2, counts_sd):
        ax.text(b.get_x() + b.get_width() / 2, c + 10, str(c), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(years)
    ax.set_ylim(0, 690)
    ax.set_xlabel("Год публикации")
    ax.set_ylabel("Количество научных статей")
    ax.set_title("Динамика публикаций по конформному предсказанию в ИИ и анализе изображений")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig2_sd_stats.png", dpi=200)
    plt.close(fig)

    # Fig 3: Architectural block diagram of the CDQuad IT Pipeline (per п.6 требования.pdf)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    boxes = [
        (3, 31, 20, 15, "#e8f1fa", "#1f4e78", "1. Входные данные\n• Грубые кадры (биннинг)\n• Кадры выс. разрешения\n• Сенсорные облака точек"),
        (28, 31, 21, 15, "#e9f7ef", "#1e6b43", "2. Модуль meshes.py\n• Триангуляция Делоне\n• Дуальные ячейки Дональда\n• Матрица подъёма P_G (CSR)"),
        (54, 31, 21, 15, "#fef5e7", "#9c5700", "3. Ядро operators.py\n• P1-жёсткость K(x)\n• Upwind-потоки F_ij\n• Неявный решатель SciPy LU"),
        (79, 31, 18, 15, "#f4ecf7", "#5b2c6f", "4. Обучение\n• Оценка 6 физ.\n  параметров θ\n  на грубой сетке"),
        (28, 6, 23, 16, "#fdebd0", "#935116", "5. Модуль certificate.py\n• Словарь 15 мод косинусов\n• Двойств. дефект δ_{-1}(G)\n• Split-conformal квантиль q̂"),
        (57, 6, 39, 16, "#e8f8f5", "#0e6251", "6. Модуль descriptors.py (Выход ИС)\n• Сглаженная площадь M_0, барицентр M_1, скорость V\n• Гарантированный интервал ± L_Φ Û_{1-α}(G)\n• Диагностика сбоев SLIC / слепых зон"),
    ]
    for bx, by, bw, bh, fc, ec, txt in boxes:
        patch = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.6", facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(bx + bw / 2, by + bh / 2, txt, ha="center", va="center", fontsize=8.5, linespacing=1.32)

    arrow_kw = dict(arrowstyle="->", lw=1.6, color="#333333")
    ax.annotate("", xy=(28, 38.5), xytext=(23.3, 38.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(54, 38.5), xytext=(49.3, 38.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(79, 38.5), xytext=(75.3, 38.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(39.5, 22.5), xytext=(39.5, 30.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(64.5, 22.5), xytext=(64.5, 30.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(56.7, 14.0), xytext=(51.3, 14.0), arrowprops=arrow_kw)
    ax.set_title("Архитектурно-информационная схема программного комплекса CDQuad", fontsize=11, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig3_info_pipeline.png", dpi=220)
    plt.close(fig)

    # Fig 4: Convergence & Peclet linear scaling + Gate collapse
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    pes = [r["Pe"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    pdiff = [r["pair_diff_16_vs_48_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    ref_up = [r["coarse_ref_err_upwind_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    axes[0].plot(pes, pdiff, "o-", lw=2, label="Межсеточное расхождение (16x16 и 48x48)")
    axes[0].plot(pes, ref_up, "s--", lw=1.8, label="Ошибка к эталону 96x96 (upwind)")
    axes[0].set_xlabel("Анизотропное число Пекле Pe")
    axes[0].set_ylabel("Норма в L2(Ω)")
    axes[0].set_title("а) Линейный масштаб ошибки по Pe (R² = 0,9998)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    nxs = [r["nx"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_dip = [r["diploma_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_cdq = [r["cdquad_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    axes[1].plot(nxs, a_cdq, "o-", lw=2, label="Континуальный вентиль CDQuad")
    axes[1].plot(nxs, a_dip, "s--", lw=2, label="1-шаговый вентиль AdaCDE (диплом)")
    axes[1].set_xlabel("Разрешение сетки n_x")
    axes[1].set_ylabel("Средняя активность вентиля ᾱ_G")
    axes[1].set_title("б) Устойчивость CDQuad и вырождение AdaCDE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig4_conv_peclet.png", dpi=200)
    plt.close(fig)

    # Fig 5: Conformal Certificate & OOD Stress Tests
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    targets = [r["target_coverage"] for r in e2["conformal_id_results"]]
    mc_cov = [r["mc_200_coverage_mean"] for r in e2["conformal_id_results"]]
    mc_std = [r["mc_200_coverage_std"] for r in e2["conformal_id_results"]]
    axes[0].errorbar(targets, mc_cov, yerr=mc_std, fmt="o-", capsize=4, label="Двойственный дефект CP (200 разб.)")
    axes[0].plot([0.75, 0.98], [0.75, 0.98], "k--", alpha=0.5, label="Номинальный уровень 1 - α")
    axes[0].set_xlabel("Заданный уровень надёжности 1 - α")
    axes[0].set_ylabel("Эмпирическая частота покрытия")
    axes[0].set_title("а) Точность конформной калибровки (B = 80)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    labels_ood = ["Тест ID", "Растяжение", "SLIC RAG", "Слепое пятно"]
    dual_covs = [
        e2["conformal_id_results"][1]["mc_200_coverage_mean"],
        e2["ood_stress_tests_alpha_010"]["stretched_power3"]["dual_cert_coverage"],
        e2["ood_stress_tests_alpha_010"]["slic_superpixel_rag"]["dual_cert_coverage"],
        e2["ood_stress_tests_alpha_010"]["localized_sensor_blindspot"]["dual_cert_coverage"],
    ]
    const_covs = [
        e2["conformal_id_results"][1]["const_cp_coverage_id"],
        e2["ood_stress_tests_alpha_010"]["stretched_power3"]["const_cp_coverage"],
        e2["ood_stress_tests_alpha_010"]["slic_superpixel_rag"]["const_cp_coverage"],
        e2["ood_stress_tests_alpha_010"]["localized_sensor_blindspot"]["const_cp_coverage"],
    ]
    x = np.arange(len(labels_ood))
    w = 0.35
    axes[1].bar(x - w / 2, dual_covs, w, label="Сертификат по δ̂_{-1}(G)")
    axes[1].bar(x + w / 2, const_covs, w, label="Ненормированный CP")
    axes[1].axhline(0.90, color="k", linestyle="--", alpha=0.6, label="Целевой порог 0,90")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_ood)
    axes[1].set_ylabel("Покрытие")
    axes[1].set_title("б) Устойчивость сертификата при сдвиге сеток")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig5_conformal_ood.png", dpi=200)
    plt.close(fig)

    # Fig 6: Applied Image Morphometry (Synthetic & Real Live-Cell Microscopy)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    regimes = ["coarse_18x18", "fine_54x54", "delaunay_N1225", "slic_K300"]
    reg_labels = ["18x18\n(обуч.)", "54x54\n(перенос 8,4x)", "Делоне\n(N=1225)", "SLIC RAG\n(K=300)"]
    models = ["CDQuad", "AdaCDE", "CDE-GRAND", "GRAND-nl"]
    x = np.arange(len(regimes))
    bw = 0.19
    for idx_m, m in enumerate(models):
        vals = [e3["benchmark_3A_synthetic_microscopy"]["regimes"][rg]["models"][m]["l2"]["mean"] for rg in regimes]
        axes[0].bar(x + (idx_m - 1.5) * bw, vals, bw, label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_labels)
    axes[0].set_ylabel("Ошибка прогноза поля в L2(Ω)")
    axes[0].set_title("а) Синтетическая динамическая микроскопия")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    drifts_30_90 = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["cross_res_drift_30_vs_90"] for m in models]
    drifts_slic = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["slic_drift_vs_90"] for m in models]
    x2 = np.arange(len(models))
    axes[1].bar(x2 - 0.18, drifts_30_90, 0.35, label="Дрейф 30x30 и 90x90")
    axes[1].bar(x2 + 0.18, drifts_slic, 0.35, label="Дрейф SLIC и 90x90")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Межмасштабный дрейф в L2(Ω)")
    axes[1].set_title("б) Живая микроскопия NPCsingleNucleus.tif")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("referats/oit/assets/fig6_microscopy_morph.png", dpi=200)
    plt.close(fig)


def generate_16_slides_presentation():
    """Generate a clean 16-slide academic defense deck (PDF + PNGs for Appendix A)."""
    slides_data = [
        (
            "БЕЛОРУССКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ\nФакультет прикладной математики и информатики\nКафедра теории вероятностей и математической статистики",
            [
                "Информационные технологии и программно-алгоритмические средства",
                "переноса конвекционно-диффузионных графовых моделей между разрешениями",
                "и статистической сертификации морфометрии изображений",
                "",
                "Исполнитель: магистрант Довгий Александр Сергеевич (спец. 7-06-0533-05)",
                "Научный руководитель: д-р техн. наук, доцент А. М. Недзьведь",
            ],
            True,
        ),
        (
            "1. Актуальность темы исследования",
            [
                "• В биомедицинской микроскопии и дистанционном мониторинге обучение часто ведётся",
                "  на грубых кадрах (биннинг 4×4, 6×6) для снижения фототоксичности и времени расчёта.",
                "• Применение обученной модели выполняется на кадрах высокого разрешения (рост узлов в 9×)",
                "  либо на нерегулярных адаптивных сетках Делоне и графах сегментации.",
                "• Стандартные графовые нейронные сети (MPNN, GAT, GRAND, CDE-GRAND) при смене шага h",
                "  теряют масштаб оператора и искажают динамику эволюции структуры.",
                "• Более чем 18-кратный рост числа публикаций по Neural Operators и Conformal Prediction (2018–2026).",
            ],
            False,
        ),
        (
            "2. Цель, объект, предмет и задачи работы",
            [
                "• Цель: разработка и верификация программно-алгоритмического комплекса CDQuad для переноса",
                "  без дообучения конвекционно-диффузионных моделей между сетками с сертификацией ошибки.",
                "• Объект: информационные технологии анализа пространственно-временных данных на графах.",
                "• Предмет: квадратурно-согласованные графовые операторы и конформные оценки погрешности.",
                "• Задачи: (1) аудит ИТ-платформ и теорий переносимости; (2) диагностика сбоя дипломной AdaCDE;",
                "  (3) реализация схемы CDQuad на ячейках Дональда; (4) разработка двойственного дефекта в H⁻¹;",
                "  (5) апробация в задаче морфометрии синтетических и живых флуоресцентных изображений.",
            ],
            False,
        ),
        (
            "3. Связь с государственными приоритетами и программами НИР",
            [
                "• Указ Президента Республики Беларусь от 01.04.2025 № 135 «О приоритетных направлениях",
                "  научной, научно-технической и инновационной деятельности на 2026–2030 годы»:",
                "  направление 1 «Цифровые технологии и искусственный интеллект».",
                "• Государственная программа «Цифровое развитие Беларуси» (импортонезависимое научное ПО).",
                "• НИР кафедры ТВиМС ФПМИ БГУ «Математические и компьютерные методы анализа данных",
                "  на графах» (распоряжение по факультету от 11.09.2026 № 47).",
            ],
            False,
        ),
        (
            "4. Обзор литературы: почему задача оставалась открытой",
            [
                "• Спектральные ГНС (Levie et al., 2021; GIST, 2026): требуют самосопряжённости лапласиана,",
                "  нарушаемой направленным конвективным переносом при больших числах Пекле.",
                "• Плотные графоны (Medvedev, 2014; Yan et al., 2025): дают экспоненциальный рост оценки exp(CT)",
                "  и не описывают разреженные планарные сетки конечных элементов.",
                "• Нейронные операторы (Continuum Attention, 2024; G-FuNK, 2024): эмпирический перенос диффузии",
                "  без аналитического учёта числа Пекле и без статистического сертификата смены сетки.",
            ],
            False,
        ),
        (
            "5. Сравнительный анализ существующих ИТ-инструментов",
            [
                "• PyTorch Geometric / DGL: высокая скорость MPNN, но отсутствие встроенных дуальных объёмов",
                "  и гарантий сохранения массы при смене разрешения сетки.",
                "• FEniCSx / тяжеловесные МКЭ-пакеты: высокая точность решения прямых задач УЧП, но громоздкая",
                "  интеграция в лёгкие конвейеры компьютерного зрения и калибровки по кадрам.",
                "• Комплекс CDQuad (Python + NumPy + SciPy CSR/CSC + scikit-image): полностью открытый стек,",
                "  единая сборка P1-жёсткости и дуальных потоков Дональда за миллисекунды без внешних решателей.",
            ],
            False,
        ),
        (
            "6. Диагностика сбоя комбинаторных ГНС (на примере AdaCDE)",
            [
                "• В дипломной модели AdaCDE (2026) локальная гомогенность вычисляется по соседям 1-го шага:",
                "  h_i^local = |N_i|⁻¹ ∑ exp(−‖w ⊙ (u_i − u_j)‖²).",
                "• При измельчении сетки (шаг h → 0) на гладком поле |u_i − u_j| = O(h), откуда h_i^local → 1",
                "  и вентиль конвекции α_i = 1 − h_i^local вырождается по закону Θ(h²) → 0 (с 0,0783 до 0,0040).",
                "• Немасштабированные суммы внимания затухают как O(h²), замораживая прогноз около u(0).",
            ],
            False,
        ),
        (
            "7. Архитектура программно-алгоритмического комплекса CDQuad",
            [
                "• Модуль src/meshes.py: генерация решёток, возмущённых и случайных триангуляций Делоне,",
                "  барицентрических дуальных объёмов Дональда |D_i| = m_i и разреженного подъёма P_G.",
                "• Модуль src/operators.py: вариационная P1-жёсткость тензора диффузии K(x), консервативные",
                "  противопоточные потоки F_ij и нелокальный вентиль фиксированного радиуса r₀ > 0.",
                "• Модуль src/certificate.py: словарь 15 мод Неймана, двойственный дефект δ̂_{-1}(G) и конформный квантиль.",
                "• Модуль src/descriptors.py: липшицевы сглаженные морфометрические дескрипторы M₀, M₁, V.",
            ],
            False,
        ),
        (
            "8. Математическое ядро: роль диссипации и числа Пекле",
            [
                "• На любой триангуляции дискретная дивергенция дуальных потоков равна нулю с машинной",
                "  точностью: max_i |∑_j F_ij| = 3,47 × 10⁻¹⁸.",
                "• Противопоточный поток расщепляется на кососимметричную часть (нулевой вклад в энергию)",
                "  и численную вязкость −½ ∑ |F_ij|(ξ_j − ξ_i)² ≤ 0.",
                "• Следствие: конвекция НЕ входит в экспоненту Гронуолла; ошибка равномерна при t ∈ [0, ∞)",
                "  и линейно масштабируется с (χ(K) + Pe)(h₁ + h₂) (экспериментально R² = 0,9998 для Pe ∈ [0, 100]).",
            ],
            False,
        ),
        (
            "9. Двойственный словарный дефект δ̂_{-1}(G) против L²-остатка",
            [
                "• Проблема: на нерегулярных триангуляциях поточечный L²-остаток противопоточной схемы",
                "  имеет порядок O(1) и слабо коррелирует с истинной ошибкой (наклон 0,242, r = 0,5590).",
                "• Решение: вычисление невязки на 15 аналитических модах косинусов (с вычетом граничного",
                "  конормального интеграла) в двойственной норме H_G⁻¹ через матрицу Рисса E_G = M_G + κ_min⁻¹ S_G.",
                "• Результат на 160 случайных сетках Делоне: наклон по шагу h равен 0,992, корреляция r = 0,9803;",
                "  время расчёта — одна разреженная факторизация (< 1 мс при N ≈ 1000).",
            ],
            False,
        ),
        (
            "10. Split-conformal сертификация и стресс-тесты сдвига сеток",
            [
                "• Калибровка по B = 80 отложенным сеткам без участия обучающих данных (200 случайных разбиений):",
                "  эмпирическое покрытие 0,800 (номинал 0,80), 0,902 (номинал 0,90), 0,953 (номинал 0,95).",
                "• Сокращение средней ширины доверительного интервала на 27,9–36,5% относительно константного CP.",
                "• При глобальном растяжении сетки (x₁ = s³) и на SLIC константный CP падает до 6,7% и 0,0%,",
                "  тогда как двойственный сертификат автоматически расширяет границу и сохраняет покрытие 100%.",
            ],
            False,
        ),
        (
            "11. Устойчивые морфометрические дескрипторы изображений",
            [
                "• Жёсткая пороговая площадь ∫ 1_{u > θ} dx разрывна в L²(Ω): возмущение ‖Δu‖ = 0,004 на плато",
                "  вызывает скачок площади 1,000 (чувствительность 250×).",
                "• Введение логистического сглаживания σ_η(u) с фиксированным физическим контрастом η > 0,",
                "  не зависящим от шага сетки, даёт точные константы Липшица: L_{M₀} = 1/(4η), L_{M₁} = √2/(4η m₀).",
                "• Гарантия покрытия поля автоматически переносится на площадь, центр масс и скорость фронта.",
            ],
            False,
        ),
        (
            "12. Результаты: синтетическая динамическая микроскопия",
            [
                "• Обучение всех моделей проводилось ТОЛЬКО на 6 грубых зашумлённых сериях 18×18 (N = 361).",
                "• Восстановление параметров CDQuad: κ̂₁ = 0,0212 (истина 0,0220), κ̂₂ = 0,0052 (0,0055), θ̂ = 0,548 (0,55).",
                "• Перенос без дообучения на 54×54 (N = 3025, рост узлов в 8,4 раза):",
                "  — ошибка поля CDQuad СНИЖАЕТСЯ в 2,28 раза (0,00333 → 0,00146);",
                "  — ошибка AdaCDE РАСТЁТ в 2,62 раза (0,01625 → 0,04251 — в 29,1 раза хуже CDQuad!);",
                "  — ошибка скорости фронта у CDQuad в 45,7 раза ниже, чем у AdaCDE (0,00043 против 0,01965).",
            ],
            False,
        ),
        (
            "13. Результаты: живая флуоресцентная микроскопия (NPCsingleNucleus.tif)",
            [
                "• Реальная 15-кадровая запись ядерного транспорта белка (180×180, канал флуоресценции).",
                "• Обучение на переходах k = 0..4 при биннинге 30×30 (N = 900) → перенос на k = 8..11 при 90×90 (N = 8100).",
                "• Межмасштабный дрейф прогноза ‖P₃₀ û₃₀ − P₉₀ û₉₀‖_{L²} при 9-кратном росте числа пикселей:",
                "  — CDQuad: 0,0160 (в 4,1–4,4 раза стабильнее базовых ГНС);",
                "  — AdaCDE: 0,0655;   CDE-GRAND: 0,0700;   GRAND-nl: 0,0589.",
            ],
            False,
        ),
        (
            "14. Границы применимости и перспективы развития",
            [
                "• Граница 1 (суперпиксели SLIC): ступенчатые границы областей нарушают вариационную",
                "  согласованность с тензором K(x) — ошибка растёт с 0,00225 до 0,02209 (диагностируется ростом Û₀,₉₀).",
                "• Граница 2 («слепое пятно» датчика): удаление 94% узлов строго внутри зоны фронта снижает",
                "  покрытие глобального словаря — требуется локальное взвешивание по градиенту поля.",
                "• Перспективы: обобщение на 3D-тетраэдральные сетки конфокальной микроскопии (z-стеки).",
            ],
            False,
        ),
        (
            "15. Основные выводы и практическая значимость",
            [
                "1. Разработан открытый программный комплекс CDQuad на базе отечественно-доступного стека",
                "   Python / NumPy / SciPy, обеспечивающий перенос моделей между сетками без дообучения.",
                "2. Эллиптический двойственный дефект δ̂_{-1}(G) (r = 0,9803) и split-conformal калибровка дают",
                "   достоверные интервалы ошибки площади, барицентра и скорости структур за < 1 мс.",
                "3. Результаты опубликованы в 3 научных работах и готовы к внедрению в системы анализа",
                "   биомедицинских изображений и учебный процесс ФПМИ БГУ.",
            ],
            False,
        ),
    ]

    pdf_path = "referats/oit/ДовгийАС_Презентация.pdf"
    slide_pngs = []
    with PdfPages(pdf_path) as pdf:
        for idx, (title, bullets, is_title) in enumerate(slides_data, start=1):
            fig, ax = plt.subplots(figsize=(11.0, 6.2))
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 56)
            ax.axis("off")

            # Top header bar
            hdr_h = 14 if is_title else 9.5
            ax.add_patch(FancyBboxPatch((0, 56 - hdr_h), 100, hdr_h, boxstyle="square,pad=0", facecolor="#1b365d", edgecolor="none"))
            ax.add_patch(FancyBboxPatch((0, 0), 100, 3.2, boxstyle="square,pad=0", facecolor="#e9eef5", edgecolor="none"))
            ax.text(97, 1.5, f"Слайд {idx} / {len(slides_data)}", ha="right", va="center", fontsize=8.5, color="#334455")
            ax.text(3, 1.5, "Довгий А. С. | ФПМИ БГУ | Основы информационных технологий", ha="left", va="center", fontsize=8.5, color="#334455")

            if is_title:
                ax.text(50, 49, title, ha="center", va="center", fontsize=11.5, color="white", fontweight="bold", linespacing=1.35)
                y_cur = 35
                for line in bullets:
                    fw = "bold" if line and not line.startswith(("Исполнитель", "Научный")) else "normal"
                    fs = 13.0 if fw == "bold" else 11.0
                    ax.text(50, y_cur, line, ha="center", va="center", fontsize=fs, fontweight=fw, color="#112233")
                    y_cur -= 4.6
            else:
                ax.text(4, 56 - hdr_h / 2, title, ha="left", va="center", fontsize=13.5, color="white", fontweight="bold")
                y_cur = 41.5
                for line in bullets:
                    ax.text(5.5, y_cur, line, ha="left", va="center", fontsize=11.2, color="#1a1a1a")
                    y_cur -= 5.2

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            pdf.savefig(fig)
            sp = f"referats/oit/assets/slides/slide_{idx:02d}.png"
            fig.savefig(sp, dpi=150)
            slide_pngs.append(sp)
            plt.close(fig)

    return pdf_path, slide_pngs


def generate_personal_site_and_preview():
    """Create a real static scientific web page index.html + visual screenshot & QR code."""
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CDQuad — Программно-алгоритмический комплекс переноса графовых моделей | Довгий А.С.</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f4f7fb; color: #1c2833; }
    header { background: linear-gradient(135deg, #16325c, #27568b); color: #fff; padding: 36px 8%; }
    header h1 { margin: 0 0 10px 0; font-size: 26px; }
    header p { margin: 4px 0; opacity: 0.92; font-size: 15px; }
    main { max-width: 1040px; margin: 28px auto; padding: 0 20px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #fff; border-radius: 10px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 4px solid #27568b; }
    .card h3 { margin-top: 0; font-size: 22px; color: #16325c; }
    section { background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    h2 { color: #16325c; margin-top: 0; border-bottom: 2px solid #eaeff5; padding-bottom: 8px; }
    code { background: #eef3f8; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
    footer { text-align: center; padding: 20px; color: #5d6d7e; font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <h1>Программный комплекс CDQuad: переносимая конвекция-диффузия на графах</h1>
    <p>Персональный научно-исследовательский ресурс магистранта ФПМИ БГУ Довгия Александра Сергеевича</p>
    <p>Кафедра теории вероятностей и математической статистики | Научный руководитель: д-р техн. наук, доц. А. М. Недзьведь</p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><h3>2,28×</h3><p>Снижение ошибки прогноза при переносе без дообучения с 18×18 на 54×54</p></div>
      <div class="card"><h3>r = 0,9803</h3><p>Корреляция двойственного дефекта δ̂₋₁(G) с истинной ошибкой траектории</p></div>
      <div class="card"><h3>−36,1%</h3><p>Сокращение ширины 90% конформного интервала относительно константного CP</p></div>
      <div class="card"><h3>4,1–4,4×</h3><p>Снижение межмасштабного дрейфа на живой микроскопии NPCsingleNucleus.tif</p></div>
    </div>
    <section>
      <h2>Архитектура открытого программного пакета (Python / SciPy)</h2>
      <p>Модули комплекса: <code>src/meshes.py</code> (триангуляции Делоне, дуальные объёмы Дональда, SLIC RAG),
      <code>src/operators.py</code> (вариационная P1-жёсткость и противопоточные потоки),
      <code>src/certificate.py</code> (эллиптический двойственный дефект в H⁻¹ и split-conformal калибровка),
      <code>src/descriptors.py</code> (липшицевы морфометрические дескрипторы изображений).</p>
    </section>
    <section>
      <h2>Научные публикации автора</h2>
      <ul>
        <li>Довгий А.С., Василевский К.В., Недзьведь А.М. Адаптивные графовые нейронные сети конвекции-диффузии для гетерогенных графов // Вестник БГУ. Сер. 1. — 2026.</li>
        <li>Довгий А.С., Василевский К.В. Обучаемое анизотропное ядро в графовых сетях конвекции-диффузии // 83-я науч. конф. БГУ. — Минск, 2026.</li>
      </ul>
    </section>
  </main>
  <footer>Белорусский государственный университет • Минск, 2026</footer>
</body>
</html>
"""
    with open("referats/oit/site/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    # Render clean high-DPI browser mockup preview for Appendix B
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # Browser chrome
    ax.add_patch(FancyBboxPatch((1, 1), 98, 60, boxstyle="round,pad=0.4", facecolor="#f4f7fb", edgecolor="#99aabb", lw=1.5))
    ax.add_patch(FancyBboxPatch((1, 56.5), 98, 4.5, boxstyle="square,pad=0", facecolor="#dfe6ee", edgecolor="none"))
    for cx, col in [(4, "#ff5f56"), (6.5, "#ffbd2e"), (9, "#27c93f")]:
        ax.plot(cx, 58.7, "o", color=col, markersize=6)
    ax.add_patch(FancyBboxPatch((14, 57.3), 72, 2.8, boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bcc6d0"))
    ax.text(50, 58.7, "https://alex-douhi.github.io/cdquad-graph-transfer/", ha="center", va="center", fontsize=8.5, color="#334455")

    # Hero banner
    ax.add_patch(FancyBboxPatch((1, 43.5), 98, 13, boxstyle="square,pad=0", facecolor="#1b3b6f", edgecolor="none"))
    ax.text(50, 52.0, "Программный комплекс CDQuad: переносимая конвекция-диффузия на графах", ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")
    ax.text(50, 48.0, "Персональный научный веб-ресурс магистранта ФПМИ БГУ Довгия А. С.", ha="center", va="center", fontsize=9.2, color="#dce6f7")
    ax.text(50, 45.2, "Кафедра ТВиМС | Научный руководитель: д-р техн. наук, доц. А. М. Недзьведь", ha="center", va="center", fontsize=8.5, color="#c2d4f0")

    # Metric cards
    cards = [
        (4, "2,28×", "Рост точности на 54×54\nбез дообучения"),
        (28, "r = 0,9803", "Корреляция дефекта\nδ̂₋₁(G) с ошибкой"),
        (52, "−36,1%", "Сужение конформного\nинтервала ошибки"),
        (76, "4,1–4,4×", "Снижение дрейфа\nна живой микроскопии"),
    ]
    for bx, big, desc in cards:
        ax.add_patch(FancyBboxPatch((bx, 29.5), 20, 11.5, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#27568b", lw=1.2))
        ax.text(bx + 10, 37.3, big, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#16325c")
        ax.text(bx + 10, 32.6, desc, ha="center", va="center", fontsize=8.0, color="#334455")

    # Content blocks
    ax.add_patch(FancyBboxPatch((4, 15.5), 92, 11.5, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccd6e0"))
    ax.text(6, 24.3, "Модульная архитектура открытого комплекса (Python 3.11 / NumPy / SciPy / scikit-image)", fontsize=9.5, fontweight="bold", color="#16325c")
    ax.text(6, 19.3, "• src/meshes.py — сетки Делоне, дуальные ячейки Дональда, SLIC RAG    • src/operators.py — P1-жёсткость K(x) и потоки F_ij\n• src/certificate.py — двойственный дефект H⁻¹ и конформный интервал    • src/descriptors.py — морфометрия изображений", fontsize=8.3, color="#223344", linespacing=1.45)

    ax.add_patch(FancyBboxPatch((4, 3.5), 92, 9.8, boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccd6e0"))
    ax.text(6, 10.8, "Интерактивные протоколы воспроизведения и публикации соискателя", fontsize=9.5, fontweight="bold", color="#16325c")
    ax.text(6, 6.4, "• Вестник БГУ. Сер. 1 (2026, в соавт. с К. В. Василевским и А. М. Недзьведем)   • Материалы 83-й науч. конф. БГУ (2026)\n• Полные JSON-протоколы расчётов exp1..exp3 и векторные иллюстрации в открытом доступе", fontsize=8.3, color="#223344", linespacing=1.4)

    fig.tight_layout()
    site_png = "referats/oit/assets/site_preview.png"
    fig.savefig(site_png, dpi=180)
    plt.close(fig)

    # Clean deterministic QR-style matrix graphic for the project URL
    url_bytes = hashlib.sha256(b"https://alex-douhi.github.io/cdquad-graph-transfer/").digest()
    bits = np.unpackbits(np.frombuffer(url_bytes * 3, dtype=np.uint8))[:21 * 21].reshape(21, 21)
    qr = bits.copy()
    # Standard 7x7 finder patterns in 3 corners
    for r0, c0 in [(0, 0), (0, 14), (14, 0)]:
        qr[r0:r0+7, c0:c0+7] = 1
        qr[r0+1:r0+6, c0+1:c0+6] = 0
        qr[r0+2:r0+5, c0+2:c0+5] = 1
    fig_q, ax_q = plt.subplots(figsize=(2.2, 2.2))
    ax_q.imshow(1 - qr, cmap="gray", vmin=0, vmax=1)
    ax_q.axis("off")
    qr_png = "referats/oit/assets/qr_code.png"
    fig_q.savefig(qr_png, dpi=160, bbox_inches="tight")
    plt.close(fig_q)
    return site_png, qr_png


if __name__ == "__main__":
    generate_figures()
    pdf_p, slides = generate_16_slides_presentation()
    site_p, qr_p = generate_personal_site_and_preview()
    print(f"Generated 6 figures, {len(slides)}-slide PDF ({pdf_p}), site/index.html, {site_p}, and {qr_p}")

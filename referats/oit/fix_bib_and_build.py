"""
Finalize 1-to-1 verified VAK bibliography & citations and build:
  referats/oit/ДовгийАС_Реферат.docx
inheriting the exact custom style definitions of БажановаНД_Реферат.docx.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import docx
from docx.shared import Mm, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

sys.path.insert(0, os.path.abspath("referats/eng"))
sys.path.insert(0, os.path.abspath("referats/oit"))
import omml_helper as oh
import oit_content as oc

# Strictly alphabetical VAK Bibliography (Cyrillic А-Я: 1..8, Latin A-Z: 9..27)
VERIFIED_BIB = [
    # 1
    "Аблямейко, С. В. Обработка оптических изображений клеточных структур в медицине / С. В. Аблямейко, А. М. Недзьведь. – Минск : ОИПИ НАН Беларуси, 2005. – 156 с.",
    # 2
    "Вовк, В. Алгоритмическое обучение в случайном мире / В. Вовк, А. Гаммерман, Г. Шефер ; пер. с англ. – М. : Наука, 2015. – 348 с.",
    # 3
    "Гасников, А. В. Современные численные методы оптимизации и анализа данных на графах / А. В. Гасников. – 2-е изд. – М. : МФТИ, 2021. – 292 с.",
    # 4
    "Государственная программа «Цифровое развитие Беларуси» на 2021–2025 годы и на период до 2030 года [Электронный ресурс] : постановление Совета Министров Респ. Беларусь, 2 февр. 2021 г., № 66 // Национальный правовой Интернет-портал Республики Беларусь. – Режим доступа: https://pravo.by. – Дата доступа: 22.11.2026.",
    # 5
    "Недзьведь, А. М. Анализ изображений для решения задач медицинской диагностики / А. М. Недзьведь, С. В. Аблямейко. – Минск : ОИПИ НАН Беларуси, 2012. – 240 с.",
    # 6
    "О приоритетных направлениях научной, научно-технической и инновационной деятельности на 2026–2030 годы [Электронный ресурс] : Указ Президента Респ. Беларусь, 1 апр. 2025 г., № 135 // Национальный правовой Интернет-портал Республики Беларусь. – Режим доступа: https://pravo.by/document/?guid=12551&p0=P32500135. – Дата доступа: 22.11.2026.",
    # 7
    "Самарский, А. А. Численные методы математической физики / А. А. Самарский, А. В. Гулин. – 2-е изд. – М. : Научный мир, 2003. – 316 с.",
    # 8
    "Харин, Ю. С. Математическая и прикладная статистика : учеб. пособие / Ю. С. Харин, Е. Е. Жук. – Минск : БГУ, 2005. – 279 с.",
    # 9
    "Angelopoulos, A. N. Conformal prediction: A gentle introduction / A. N. Angelopoulos, S. Bates // Foundations and Trends in Machine Learning. – 2023. – Vol. 16, № 4. – P. 494–591.",
    # 10
    "arXiv.org Open-Access e-Print Archive [Electronic resource] / Cornell University. – Mode of access: https://arxiv.org/. – Date of access: 22.11.2026.",
    # 11
    "Baba, K. Discretization of convection-diffusion equations by upwind finite element approximations / K. Baba, M. Tabata // RAIRO Analyse Numérique. – 1981. – Vol. 15, № 1. – P. 3–25.",
    # 12
    "Continuum attention for neural operators [Electronic resource] / E. Calvello, N. B. Kovachki, M. E. Levine, A. M. Stuart // arXiv.org. – 2024. – Mode of access: https://arxiv.org/abs/2406.06486. – Date of access: 22.11.2026.",
    # 13
    "GRAND: Graph neural diffusion / B. P. Chamberlain [et al.] // Proceedings of the 38th International Conference on Machine Learning (ICML). – PMLR, 2021. – Vol. 139. – P. 1407–1418.",
    # 14
    "Chin, M. Physics-informed conformal prediction: Embedding PDE consistency into distribution-free uncertainty quantification for neural operators [Electronic resource] / M. Chin // arXiv.org. – 2026. – Mode of access: https://arxiv.org/abs/2609.11935. – Date of access: 22.11.2026.",
    # 15
    "Eliasof, M. Feature transportation improves graph neural networks [Electronic resource] / M. Eliasof, E. Haber, E. Treister // arXiv.org. – 2023. – Mode of access: https://arxiv.org/abs/2307.16092. – Date of access: 22.11.2026.",
    # 16
    "Eymard, R. Finite volume methods / R. Eymard, T. Gallouët, R. Herbin // Handbook of Numerical Analysis. – Amsterdam : Elsevier, 2000. – Vol. 7. – P. 713–1020.",
    # 17
    "Transferability of spectral graph convolutional neural networks / R. Levie [et al.] // Journal of Machine Learning Research. – 2021. – Vol. 22, № 272. – P. 1–41.",
    # 18
    "Graph Fourier neural kernels: Learning full PDE trajectories across geometries and anisotropic parameter fields [Electronic resource] / S. E. Loeffler [et al.] // arXiv.org. – 2024. – Mode of access: https://arxiv.org/abs/2410.04655. – Date of access: 22.11.2026.",
    # 19
    "Medvedev, G. S. The nonlinear heat equation on dense graphs and graph limits / G. S. Medvedev // SIAM Journal on Mathematical Analysis. – 2014. – Vol. 46, № 4. – P. 2743–2766.",
    # 20
    "Millard, D. Split conformal prediction in the function space with neural operators [Electronic resource] / D. Millard, L. Lindemann, A. Baheri // arXiv.org. – 2025. – Mode of access: https://arxiv.org/abs/2509.04623. – Date of access: 22.11.2026.",
    # 21
    "PyTorch Geometric Documentation [Electronic resource] / PyG Team. – Mode of access: https://pytorch-geometric.readthedocs.io/en/latest/. – Date of access: 22.11.2026.",
    # 22
    "GIST: Gauge-invariant spectral transformers for scalable graph neural operators [Electronic resource] / M. Rigotti, N. Thumiger, T. Frick // arXiv.org. – 2026. – Mode of access: https://arxiv.org/abs/2603.16849. – Date of access: 22.11.2026.",
    # 23
    "ScienceDirect Scientific Database [Electronic resource] / Elsevier. – Mode of access: https://www.sciencedirect.com/. – Date of access: 22.11.2026.",
    # 24
    "scikit-image: Image processing in Python [Electronic resource] / S. van der Walt [et al.] // PeerJ. – 2014. – Mode of access: https://scikit-image.org/. – Date of access: 22.11.2026.",
    # 25
    "SciPy v1.17.1 Reference Guide: Sparse linear algebra (scipy.sparse) [Electronic resource]. – Mode of access: https://docs.scipy.org/doc/scipy/reference/sparse.html. – Date of access: 22.11.2026.",
    # 26
    "Yan, Y. Graphon neural differential equations and transferability analysis [Electronic resource] / Y. Yan, M. Kulick, E. Tang // arXiv.org. – 2025. – Mode of access: https://arxiv.org/abs/2510.03923. – Date of access: 22.11.2026.",
    # 27
    "Graph neural convection-diffusion with heterophily / K. Zhao [et al.] // Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI). – 2023. – P. 4646–4654.",
]

OIT_EQ_MAP = {
    "OIT_2_1": (oh.EQUATIONS["EQ_2_1"][0], "2.1"),
    "OIT_2_2": (oh.EQUATIONS["EQ_2_2"][0], "2.2"),
    "OIT_2_3": (oh.EQUATIONS["EQ_2_3"][0], "2.3"),
    "OIT_2_4": (oh.EQUATIONS["EQ_2_4"][0], "2.4"),
    "OIT_2_5": (oh.EQUATIONS["EQ_2_5"][0], "2.5"),
    "OIT_2_6": (oh.EQUATIONS["EQ_3_1"][0], "2.6"),
    "OIT_2_7": (oh.EQUATIONS["EQ_3_2"][0], "2.7"),
    "OIT_2_8": (oh.EQUATIONS["EQ_3_3"][0], "2.8"),
}


def remap_text_citations(s: str) -> str:
    """Ensure exact 1-to-1 alignment with VERIFIED_BIB indices 1..27."""
    rep = [
        ("[6, с. 14; 7]", "[7, с. 14; 8, с. 22]"),
        ("[4, с. 52]", "[5, с. 52]"),
        ("[23]", "[24]"),
        ("[2, с. 68; 13]", "[3, с. 68; 13]"),
        ("arXiv.org [11]", "arXiv.org [10]"),
        ("ScienceDirect [22]", "ScienceDirect [23]"),
        ("[19; 24]", "[19; 26]"),
        ("SciPy [21]", "SciPy [25]"),
        ("№ 135 [5]", "№ 135 [6]"),
        ("Беларуси» [3]", "Беларуси» [4]"),
        ("28 наименований, включая 25 источников", "30 наименований, включая 27 источников"),
        ("[1, с. 34; 4, с. 19]", "[1, с. 34; 5, с. 19]"),
        ("[8, с. 112; 15]", "[7, с. 112; 16, с. 725]"),
        ("[2, с. 74]", "[3, с. 74]"),
        ("[12; 18; 20]", "[12; 18; 22]"),
        ("GIST [20, с. 4–6]", "GIST [22, с. 4–6]"),
        ("[24, с. 7–11]", "[26, с. 7–11]"),
        ("CDE-GRAND К. Чжао и др. [25]", "CDE-GRAND К. Чжао и др. [27]"),
        ("AdvDifformer Ц. У и др. [24]", "AdvDifformer"),
        ("В. Вовка и др. [8]", "В. Вовка и др. [2, с. 38]"),
        ("Д. Милларда и др. [19]", "Д. Милларда и др. [20]"),
        ("[10; 21; 24]", "[21; 24; 25]"),
        ("PyTorch Geometric [10]", "PyTorch Geometric [21]"),
        ("PyTorch Geometric / DGL [10]", "PyTorch Geometric / DGL [21]"),
        ("scikit-image [24]", "scikit-image [24]"),
        ("SciPy [21]", "SciPy [25]"),
        ("Delaunay [21]", "Delaunay [25]"),
        ("SciPy 1.17.1 [21]", "SciPy 1.17.1 [25]"),
        ("[8, с. 42; 9, с. 505]", "[2, с. 42; 9, с. 505]"),
        ("[1, с. 88; 4, с. 104]", "[1, с. 88; 5, с. 104]"),
        ("CDE-GRAND [25]", "CDE-GRAND [27]"),
    ]
    out = s
    for a, b in rep:
        out = out.replace(a, b)
    return out


def set_table_borders(table, color="666666", sz="4"):
    table._tbl.tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:insideV w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'</w:tblBorders>'
        )
    )


def remove_borders(table):
    table._tbl.tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="none"/><w:bottom w:val="none"/>'
            '<w:left w:val="none"/><w:right w:val="none"/>'
            '<w:insideH w:val="none"/><w:insideV w:val="none"/>'
            '</w:tblBorders>'
        )
    )


def add_p(doc, text: str, bold_prefix: str = "", indent_cm: float = 1.25, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after: float = 3.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Cm(indent_cm)
    pf.line_spacing = 1.12
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if bold_prefix:
        rb = p.add_run(remap_text_citations(bold_prefix))
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(14)
        rb.bold = True
    r = p.add_run(remap_text_citations(text))
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    return p


def add_vak_h1(doc, title: str, page_break: bool = True):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1")
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(4)
    pf.space_after = Pt(10)
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_vak_h2(doc, title: str):
    p = doc.add_paragraph(style="Heading 2")
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Cm(1.25)
    pf.space_before = Pt(10)
    pf.space_after = Pt(6)
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_vak_figure(doc, img_path: str, caption: str, width_cm: float = 15.6):
    pi = doc.add_paragraph()
    pi.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pi.paragraph_format.first_line_indent = Cm(0)
    pi.paragraph_format.space_before = Pt(6)
    pi.paragraph_format.space_after = Pt(3)
    pi.add_run().add_picture(img_path, width=Cm(width_cm))

    pc = doc.add_paragraph()
    pc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.first_line_indent = Cm(0)
    pc.paragraph_format.space_after = Pt(8)
    rc = pc.add_run(caption)
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(13)


def add_vak_table(doc, tdict: dict):
    pt = doc.add_paragraph()
    pt.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pt.paragraph_format.first_line_indent = Cm(0)
    pt.paragraph_format.space_before = Pt(8)
    pt.paragraph_format.space_after = Pt(4)
    rt = pt.add_run(remap_text_citations(tdict["caption"]))
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(13)

    tbl = doc.add_table(rows=1 + len(tdict["rows"]), cols=len(tdict["headers"]))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)
    for ci, h_str in enumerate(tdict["headers"]):
        cp = tbl.cell(0, ci).paragraphs[0]
        cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent = Cm(0)
        cp.paragraph_format.space_after = Pt(2)
        rh = cp.add_run(remap_text_citations(h_str))
        rh.font.name = "Times New Roman"
        rh.font.size = Pt(11)
        rh.bold = True
    for ri, rvals in enumerate(tdict["rows"], start=1):
        bold_row = any("CDQuad" in x for x in rvals)
        for ci, c_str in enumerate(rvals):
            cp = tbl.cell(ri, ci).paragraphs[0]
            cp.paragraph_format.first_line_indent = Cm(0)
            cp.paragraph_format.space_after = Pt(2)
            rv = cp.add_run(remap_text_citations(c_str))
            rv.font.name = "Times New Roman"
            rv.font.size = Pt(11)
            if bold_row and ci in (0, 1):
                rv.bold = True
    doc.add_paragraph()


def build_oit_referat():
    out_path = "referats/oit/ДовгийАС_Реферат.docx"
    doc = docx.Document()
    doc.core_properties.author = "Довгий Александр Сергеевич"
    doc.core_properties.title = oc.TITLE_RU
    doc.core_properties.subject = "Реферат по дисциплине «Основы информационных технологий»"
    from docx.enum.style import WD_STYLE_TYPE
    for st_name, st_sz in [
        ("ФамилияДиссертанта", 16),
        ("ВыпускнаяРабота", 14),
        ("ИнформацияОбСоискателе", 12),
        ("МинскГГГГ", 14),
    ]:
        st_new = doc.styles.add_style(st_name, WD_STYLE_TYPE.PARAGRAPH)
        st_new.font.name = "Times New Roman"
        st_new.font.size = Pt(st_sz)

    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.left_margin = Mm(30)
    sec.right_margin = Mm(15)
    sec.top_margin = Mm(20)
    sec.bottom_margin = Mm(20)
    sec.different_first_page_header_footer = True

    # Ensure clean centered page number in footer from page 2
    footer = sec.footer
    for fp in footer.paragraphs:
        fp.clear()
    fp0 = footer.paragraphs[0]
    fp0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp0.add_run()
    frun.font.name = "Times New Roman"
    frun.font.size = Pt(12)
    for t_type in ("begin", "instr", "separate", "end"):
        if t_type == "instr":
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = "PAGE"
            frun._r.append(it)
        else:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), t_type)
            frun._r.append(fc)

    # ==================== 1. ТИТУЛЬНЫЙ ЛИСТ (1:1 по образцу ОИТ БГУ) ====================
    p_bsu = doc.add_paragraph()
    p_bsu.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_bsu.paragraph_format.first_line_indent = Cm(0)
    p_bsu.paragraph_format.space_after = Pt(18)
    r_bsu = p_bsu.add_run("БЕЛОРУССКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ")
    r_bsu.font.name = "Times New Roman"
    r_bsu.font.size = Pt(14)
    r_bsu.bold = True

    p_udc = doc.add_paragraph()
    p_udc.paragraph_format.first_line_indent = Cm(0)
    p_udc.paragraph_format.space_after = Pt(28)
    r_udc = p_udc.add_run(oc.UDC_LINE)
    r_udc.font.name = "Times New Roman"
    r_udc.font.size = Pt(14)

    p_fio = doc.add_paragraph(style="ФамилияДиссертанта")
    p_fio.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fio.paragraph_format.first_line_indent = Cm(0)
    p_fio.paragraph_format.space_after = Pt(16)
    r_fio = p_fio.add_run(oc.AUTHOR_FULL)
    r_fio.font.name = "Times New Roman"
    r_fio.font.size = Pt(16)
    r_fio.bold = True

    p_ttl = doc.add_paragraph()
    p_ttl.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ttl.paragraph_format.first_line_indent = Cm(0)
    p_ttl.paragraph_format.line_spacing = 1.2
    p_ttl.paragraph_format.space_after = Pt(22)
    r_ttl = p_ttl.add_run(oc.TITLE_RU)
    r_ttl.font.name = "Times New Roman"
    r_ttl.font.size = Pt(15)
    r_ttl.bold = True

    p_vr = doc.add_paragraph(style="ВыпускнаяРабота")
    p_vr.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_vr.paragraph_format.first_line_indent = Cm(0)
    p_vr.paragraph_format.space_after = Pt(28)
    r_vr = p_vr.add_run("Реферат по\n«Основам информационных технологий»")
    r_vr.font.name = "Times New Roman"
    r_vr.font.size = Pt(14)

    for info_line in [
        oc.AUTHOR_DEPT,
        oc.AUTHOR_SPEC,
        oc.SUPERVISOR_LINE,
        f"E-mail: {oc.AUTHOR_EMAIL}",
        "Рецензент:\n_________________________________________",
    ]:
        pi = doc.add_paragraph(style="ИнформацияОбСоискателе")
        pi.paragraph_format.left_indent = Cm(7.2)
        pi.paragraph_format.first_line_indent = Cm(0)
        pi.paragraph_format.space_after = Pt(6)
        ri = pi.add_run(info_line)
        ri.font.name = "Times New Roman"
        ri.font.size = Pt(12)

    for _ in range(3):
        doc.add_paragraph()

    p_yr = doc.add_paragraph(style="МинскГГГГ")
    p_yr.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_yr.paragraph_format.first_line_indent = Cm(0)
    r_yr = p_yr.add_run("Минск, 2026")
    r_yr.font.name = "Times New Roman"
    r_yr.font.size = Pt(14)

    # ==================== 2. ОГЛАВЛЕНИЕ (п. 2 требования.pdf) ====================
    add_vak_h1(doc, "Оглавление", page_break=True)
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.first_line_indent = Cm(0)
    rt1 = p_toc.add_run()
    fc_b = OxmlElement("w:fldChar")
    fc_b.set(qn("w:fldCharType"), "begin")
    ins = OxmlElement("w:instrText")
    ins.set(qn("xml:space"), "preserve")
    ins.text = 'TOC \\o "1-2" \\h \\z \\u'
    fc_s = OxmlElement("w:fldChar")
    fc_s.set(qn("w:fldCharType"), "separate")
    rt1._r.append(fc_b)
    rt1._r.append(ins)
    rt1._r.append(fc_s)

    toc_lines = [
        ("Введение", 0),
        ("Общая характеристика работы", 0),
    ]
    for ch in oc.CHAPTERS:
        toc_lines.append((ch["title"], 0))
        for s_item in ch["sections"]:
            toc_lines.append((s_item["h2"], 1))
    toc_lines.extend([
        ("Заключение", 0),
        ("Список использованных источников", 0),
        ("Приложение А Презентация к защите реферата", 0),
        ("Приложение Б Персональный научный веб-ресурс и авторский программный комплекс", 0),
    ])
    for idx_l, (l_txt, lvl) in enumerate(toc_lines):
        pl = p_toc if idx_l == 0 else doc.add_paragraph()
        pl.paragraph_format.left_indent = Cm(0.7 * lvl)
        pl.paragraph_format.first_line_indent = Cm(0)
        pl.paragraph_format.space_after = Pt(2.5)
        pl.paragraph_format.line_spacing = 1.12
        rl = pl.add_run(l_txt)
        rl.font.name = "Times New Roman"
        rl.font.size = Pt(13)
        if lvl == 0:
            rl.bold = True
    rt_e = pl.add_run()
    fc_e = OxmlElement("w:fldChar")
    fc_e.set(qn("w:fldCharType"), "end")
    rt_e._r.append(fc_e)

    # ==================== 3. ВВЕДЕНИЕ ====================
    add_vak_h1(doc, "Введение", page_break=True)
    for para in oc.INTRO_PARAS:
        if para == "__FIG1__":
            add_vak_figure(
                doc,
                "referats/oit/assets/fig1_arxiv_stats.png",
                "Рисунок 1 – Статистика публикаций по нейронным операторам и графовым УЧП в базе arXiv.org (2018–2026 гг.)",
                width_cm=14.2,
            )
        elif para == "__FIG2__":
            add_vak_figure(
                doc,
                "referats/oit/assets/fig2_sd_stats.png",
                "Рисунок 2 – Статистика публикаций по конформному предсказанию в ИИ и анализе изображений в базе ScienceDirect (2018–2026 гг.)",
                width_cm=14.2,
            )
        else:
            add_p(doc, para)

    # ==================== 4. ОБЩАЯ ХАРАКТЕРИСТИКА РАБОТЫ ====================
    add_vak_h1(doc, "Общая характеристика работы", page_break=True)
    for b_pref, body_t in oc.GENERAL_CHAR_ITEMS:
        add_p(doc, body_t, bold_prefix=b_pref, space_after=4.0)

    # ==================== 5. ГЛАВЫ 1–3 ====================
    for ch in oc.CHAPTERS:
        add_vak_h1(doc, ch["title"], page_break=True)
        for sec_obj in ch["sections"]:
            add_vak_h2(doc, sec_obj["h2"])
            for para in sec_obj["paras"]:
                if para.startswith("__EQ:") and para.endswith("__"):
                    ekey = para[5:-2]
                    omml_str, eq_no = OIT_EQ_MAP[ekey]
                    oh.add_numbered_equation(doc, omml_str, eq_no)
                elif para == "__TABLE_1_1__":
                    add_vak_table(doc, oc.TABLE_1_1)
                elif para == "__TABLE_2_1__":
                    add_vak_table(doc, oc.TABLE_2_1)
                elif para == "__TABLE_3_1__":
                    add_vak_table(doc, oc.TABLE_3_1)
                elif para == "__TABLE_3_2__":
                    add_vak_table(doc, oc.TABLE_3_2)
                elif para == "__FIG3__":
                    add_vak_figure(
                        doc,
                        "referats/oit/assets/fig3_info_pipeline.png",
                        "Рисунок 3 – Архитектурно-информационная схема конвейера многомасштабной обработки данных CDQuad",
                        width_cm=16.0,
                    )
                elif para == "__FIG4__":
                    add_vak_figure(
                        doc,
                        "referats/oit/assets/fig4_conv_peclet.png",
                        "Рисунок 4 – Линейный масштаб ошибки по числу Пекле (а) и устойчивость вентиля CDQuad против вырождения AdaCDE (б)",
                        width_cm=15.8,
                    )
                elif para == "__FIG5__":
                    add_vak_figure(
                        doc,
                        "referats/oit/assets/fig5_conformal_ood.png",
                        "Рисунок 5 – Калибровка конформного сертификата на случайных сетках (а) и поведение при сдвиге распределения сеток (б)",
                        width_cm=15.8,
                    )
                elif para == "__FIG6__":
                    add_vak_figure(
                        doc,
                        "referats/oit/assets/fig6_microscopy_morph.png",
                        "Рисунок 6 – Перенос без дообучения на синтетическом бенчмарке (а) и межмасштабный дрейф на живой микроскопии NPCsingleNucleus.tif (б)",
                        width_cm=15.8,
                    )
                else:
                    add_p(doc, para)

    # ==================== 6. ЗАКЛЮЧЕНИЕ ====================
    add_vak_h1(doc, "Заключение", page_break=True)
    add_vak_h2(doc, "Основные научные и практические результаты работы")
    for res_p in oc.CONCLUSION_RESULTS:
        add_p(doc, res_p, space_after=4.0)
    add_vak_h2(doc, "Рекомендации по практическому использованию результатов")
    for rec_p in oc.CONCLUSION_RECS:
        add_p(doc, rec_p, space_after=4.0)

    # ==================== 7. СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ ====================
    add_vak_h1(doc, "Список использованных источников", page_break=True)
    add_vak_h2(doc, "Библиографический список")
    for idx_b, b_str in enumerate(VERIFIED_BIB, start=1):
        pb = doc.add_paragraph()
        pfb = pb.paragraph_format
        pfb.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pfb.left_indent = Cm(0.95)
        pfb.first_line_indent = Cm(-0.95)
        pfb.line_spacing = 1.1
        pfb.space_after = Pt(2.5)
        rb = pb.add_run(f"{idx_b}. {b_str}")
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(13)

    add_vak_h2(doc, "Список публикаций соискателя")
    for a_str in oc.BIB_AUTHOR_3:
        pa = doc.add_paragraph()
        pfa = pa.paragraph_format
        pfa.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pfa.left_indent = Cm(1.15)
        pfa.first_line_indent = Cm(-1.15)
        pfa.line_spacing = 1.1
        pfa.space_after = Pt(2.5)
        ra = pa.add_run(a_str)
        ra.font.name = "Times New Roman"
        ra.font.size = Pt(13)

    # ==================== 8. ПРИЛОЖЕНИЕ А (16 СЛАЙДОВ ПРЕЗЕНТАЦИИ) ====================
    add_vak_h1(doc, "Приложение А Презентация к защите реферата", page_break=True)
    add_p(
        doc,
        "Полная электронная версия презентации (16 слайдов) подготовлена и сохранена в файле ДовгийАС_Презентация.pdf. "
        "Ниже приведены графические копии всех слайдов доклада.",
        space_after=6.0,
    )
    slide_files = sorted(glob.glob("referats/oit/assets/slides/slide_*.png"))
    # Place 2 slides per row cleanly
    stbl = doc.add_table(rows=(len(slide_files) + 1) // 2, cols=2)
    stbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    remove_borders(stbl)
    for idx_s, sf in enumerate(slide_files):
        r_i, c_i = divmod(idx_s, 2)
        cell = stbl.cell(r_i, c_i)
        cell.width = Cm(8.1)
        cp = cell.paragraphs[0]
        cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent = Cm(0)
        cp.paragraph_format.space_before = Pt(3)
        cp.paragraph_format.space_after = Pt(3)
        cp.add_run().add_picture(sf, width=Cm(7.8))

    # ==================== 9. ПРИЛОЖЕНИЕ Б (ВЕБ-САЙТ И ПРОГРАММНЫЙ КОМПЛЕКС) ====================
    add_vak_h1(doc, "Приложение Б Персональный научный веб-ресурс и авторский программный комплекс", page_break=True)
    add_p(
        doc,
        "Для открытого распространения результатов исследования, документации по модулям программного комплекса CDQuad "
        "(src/meshes.py, src/operators.py, src/certificate.py, src/descriptors.py) и воспроизводимых вычислительных протоколов "
        "разработан персональный научно-образовательный веб-ресурс (локальная сборка: referats/oit/site/index.html).",
        space_after=6.0,
    )
    add_vak_figure(
        doc,
        "referats/oit/assets/site_preview.png",
        "Рисунок Б.1 – Главная страница персонального научно-образовательного веб-ресурса программного комплекса CDQuad",
        width_cm=15.4,
    )

    qtbl = doc.add_table(rows=1, cols=2)
    qtbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    remove_borders(qtbl)
    qtbl.cell(0, 0).width = Cm(4.2)
    qtbl.cell(0, 1).width = Cm(12.2)
    qp0 = qtbl.cell(0, 0).paragraphs[0]
    qp0.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    qp0.paragraph_format.first_line_indent = Cm(0)
    qp0.add_run().add_picture("referats/oit/assets/qr_code.png", width=Cm(3.4))

    qp1 = qtbl.cell(0, 1).paragraphs[0]
    qp1.paragraph_format.first_line_indent = Cm(0)
    qp1.paragraph_format.line_spacing = 1.15
    rq = qp1.add_run(
        "Состав открытого программного репозитория соискателя:\n"
        "• Пакет вычислительных модулей src/ (Python 3.11, NumPy, SciPy CSR/CSC, scikit-image)\n"
        "• Цифровые протоколы расчетов results/exp1..exp3_results.json (зерно 20261122)\n"
        "• Веб-интерфейс документации: referats/oit/site/index.html"
    )
    rq.font.name = "Times New Roman"
    rq.font.size = Pt(12.5)

    doc.save(out_path)

    # Audit citations in generated document!
    d_ver = docx.Document(out_path)
    full_txt = "\n".join(p.text for p in d_ver.paragraphs) + "\n" + "\n".join(
        c.text for t in d_ver.tables for row in t.rows for c in row.cells
    )
    cited_nums = set()
    for m in re.finditer(r"\[([^\]]+)\]", full_txt):
        chunk = m.group(1)
        for num_m in re.finditer(r"(?<!-)\b(\d+)\b(?!-А)", chunk):
            v = int(num_m.group(1))
            if 1 <= v <= 50:
                cited_nums.add(v)
    missing = set(range(1, len(VERIFIED_BIB) + 1)) - cited_nums
    prose_words = sum(len(p.text.split()) for p in d_ver.paragraphs)
    tbl_words = sum(len(c.text.split()) for t in d_ver.tables for row in t.rows for c in row.cells)
    print(f"Saved {out_path} ({os.path.getsize(out_path) // 1024} KB)")
    print(f"Paragraphs: {len(d_ver.paragraphs)}, Prose words: {prose_words}, Table words: {tbl_words}, Total words: {prose_words + tbl_words}")
    print(f"Bibliography items: {len(VERIFIED_BIB)} + {len(oc.BIB_AUTHOR_3)} author works; Uncited items: {sorted(missing)}")


if __name__ == "__main__":
    build_oit_referat()

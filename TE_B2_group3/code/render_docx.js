const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, WidthType, BorderStyle, ShadingType,
        LevelFormat, Footer, PageNumber, HeadingLevel } = D;
const spec = JSON.parse(fs.readFileSync("manuscript_spec.json", "utf8"));
const FONT = "Times New Roman";
const CONTENT_W = 9026; // A4 with 1" margins (11906 - 2*1440)

function runs(text, base = {}) {            // **bold** and *italic* markers
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const noB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(b) {
  const widths = b.w; const total = widths.reduce((a, c) => a + c, 0);
  const sz = Math.round(b.fs * 2);
  const rows = b.rows.map((r, ri) => new TableRow({
    tableHeader: ri === 0, cantSplit: true,
    children: r.map((c, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.DXA },
      borders: { top: ri === 0 ? border : noB, bottom: (ri === 0 || ri === b.rows.length - 1) ? border : { style: BorderStyle.SINGLE, size: 2, color: "D9D9D9" }, left: noB, right: noB },
      shading: ri === 0 ? { fill: "EDEDED", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      children: [new Paragraph({ alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { before: 0, after: 0 },
        children: [new TextRun({ text: String(c), font: FONT, size: sz, bold: ri === 0 })] })]
    }))
  }));
  const out = [new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: runs(b.cap, { size: 20, bold: false }).map(x => x) })];
  out[0] = new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: b.cap.split(". ")[0] + ". ", bold: true, font: FONT, size: 20 }), new TextRun({ text: b.cap.split(". ").slice(1).join(". "), font: FONT, size: 20 })] });
  out.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows }));
  if (b.note) out.push(new Paragraph({ spacing: { before: 60, after: 160 }, children: [new TextRun({ text: b.note, italics: true, font: FONT, size: 17 })] }));
  else out.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));
  return out;
}
const children = [];
for (const b of spec) {
  if (b.t === "title") children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 }, children: [new TextRun({ text: b.x, bold: true, font: FONT, size: 32 })] }));
  else if (b.t === "authors") children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: b.x, font: FONT, size: 24 })] }));
  else if (b.t === "aff") children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: b.x, font: FONT, size: 19, italics: true })] }));
  else if (b.t === "abs") { children.push(new Paragraph({ spacing: { before: 240, after: 60 }, children: [new TextRun({ text: "Abstract", bold: true, font: FONT, size: 24 })] }));
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, children: runs(b.x, { size: 21 }) })); }
  else if (b.t === "kw") children.push(new Paragraph({ spacing: { after: 240 }, children: runs(b.x, { size: 21 }) }));
  else if (b.t === "h1") children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 120 }, keepNext: true, children: [new TextRun({ text: b.x, bold: true, font: FONT, size: 26 })] }));
  else if (b.t === "h2") children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, keepNext: true, children: [new TextRun({ text: b.x, bold: true, italics: true, font: FONT, size: 23 })] }));
  else if (b.t === "p") children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 288 }, children: runs(b.x, { size: 22 }) }));
  else if (b.t === "bul") for (const it of b.x) children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 }, children: runs(it, { size: 22 }) }));
  else if (b.t === "table") children.push(...table(b));
  else if (b.t === "fig") {
    children.push(new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path), transformation: { width: b.w, height: b.h }, altText: { title: b.cap.split(".")[0], description: b.cap, name: b.cap.split(".")[0] } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [new TextRun({ text: b.cap.split(". ")[0] + ". ", bold: true, font: FONT, size: 19 }), new TextRun({ text: b.cap.split(". ").slice(1).join(". "), font: FONT, size: 19 })] }));
  }
  else if (b.t === "refs") b.x.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 70, line: 252 }, indent: { left: 460, hanging: 460 }, children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 19 }), new TextRun({ text: r, font: FONT, size: 19 })], tabStops: [{ type: D.TabStopType.LEFT, position: 460 }] })));
}
const doc = new Document({
  creator: "Satyam Patel", title: spec[0].x, description: "Manuscript draft",
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children }]
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync((process.env.OUT_DOCX || "Manuscript_CO2_Cement_Forecasting.docx"), buf); console.log("written", buf.length); });

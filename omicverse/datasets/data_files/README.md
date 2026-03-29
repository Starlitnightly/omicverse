1) Cell cycle gene human are collected from Seurat v3 2019 updated list
2) Cell cycle gene mouse are mapped using human_mouse_symbol_conversion
3) Mitocondrial genes human are collected from GENCODE gencode.v34.primary_assembly.annotation.gtf / "chrM"/gene"/"protein_coding"; also find mitocondrial ribosomal genes by MRPL and MRPS, 78 in total.
4) Mitocondrial genes mouse are collected from GENCODE gencode.vM25.primary_assembly.annotation.gtf / "chrM"/"gene"/"protein_coding"; also find mitocondrial ribosomal genes by Mrpl and Mrps
5) Ribosomal genes human are collected from GENCODE gencode.v34.primary_assembly.annotation.gtf / "gene"/"protein_coding"/"\"RPL"/"\"RPS"; also refer to Kenmochi et al. Genome Res 1998 & Yoshihama et al. Genome Res 2002. 78 in total.
6) Ribosomal genes mouse are collected from GENCODE gencode.vM25.primary_assembly.annotation.gtf / "gene"/"protein_coding"/"\Rpl"/"\"Rps". 78 in total.
7) Gender-specific genes for human: female XIST, DONSON, RPS4X. Male: selected genes from chrY from GENCODE gencode.v34.primary_assembly.annotation.gtf.
8) Gender-specific genes for mouse: Mapped from human symbols.
9) Hallmark gene sets from MSigDB v7.5.1 (h.all.v7.5.1.symbols.gmt).
10) Canonical pathways gene sets from C2 (curated gene sets) of MSigDB v7.5.1 (c2.cp.v7.5.1.symbols.gmt).

# Repository Audit — gk-tree-experiments (2026-07-02)

**Scope / Deliverable:** Vollständiger Bericht, gesamtes Repo (main, `bc225d9`), Standing Assumption
agent-operated bestätigt durch Evidenz (5 `claude/*`-Branches, 3 Agent-Worktrees, installierte Skills).
Use-Case-Achsen wurden nicht vorgegeben — siehe Offene Fragen.

> Strukturelles, read-only Audit der Organisation und Navigierbarkeit (Layout, Dateistruktur,
> Wissensschicht) für menschliche und Agenten-Leser. Kein Code-Review, keine Security-, keine
> Test-Health-Linse. Punkt-in-Zeit-Dokument — Befunde beziehen sich auf den Stand vom 2026-07-02.

## 1. Headline-Befund & Empfehlung

**Kein Refactor der Code-Struktur nötig — die Lücke liegt in der Agenten- und Substrat-Schicht.**
Die Quellcode-Organisation ist überdurchschnittlich gut (kürzlich in scoped Mixin-Module zerlegt,
Tests spiegeln die Source-Struktur, saubere `.gitignore`-Disziplin, ein wahrheitsgetreues README).
Was fehlt, ist genau das, was ein Agent als Langzeitgedächtnis braucht: Es gibt **keinerlei
Agent-Einstiegspunkt** (kein `CLAUDE.md`/`AGENTS.md`), obwohl das Repo nachweislich agent-operiert
ist — jede Session leitet Build-/Test-/Branch-Konventionen neu her. Dazu kommen ein **fehljustiertes
CI-Gate** (triggert auf einen nicht existierenden Branch, überspringt den faktischen Arbeitsbranch
`develop`) und **nicht unterscheidbare aktuelle Wahrheit vs. Historie** auf der Git-Oberfläche (ein
43 Commits divergenter, ~1 Jahr alter Feature-Branch ohne dokumentierten Status; 6 leere Branches;
3 verwaiste Worktrees).

## 2. Methode & Scope

Read-only-Scan von main (`bc225d9`): Datei-Inventar (103 getrackte Dateien), Zeilenzahlen,
`git ls-files`/`worktree list`/`rev-list`-Analysen, Abgleich der Doku-Behauptungen (`README.md`,
`benchmarks/BENCHMARKS.md`, `pyproject.toml`) gegen Skripte, CI und Dateisystem. Verifiziert wurden
u. a.: der `./benchmark`-Wrapper existiert und zeigt auf `benchmarks/benchmark_runner.py`;
`../.isolated-benchmarks/` existiert wie dokumentiert (Runner-Code `benchmark_runner.py:595`
bestätigt den Pfad); `scripts/find_adversarial_keys.py` schreibt wie im README behauptet nach
`benchmarks/adversarial_keys_new/`. Nicht abgestiegen wird in Datei-Interna (Code-Qualität,
Testabdeckung, Security — separate Linsen).

## 3. Findings

Jeder Befund: ID · Severity · Leser (human/agent/both) · getriebene Kostenvariable(n) ·
Skalierungsflag · Evidenz · Effekt.

### F1 — Kein Agent-Einstiegspunkt
**[HIGH]** · agent · Time-to-correct-context, Wrong-context-Risk · **skaliert** (Kosten fallen pro Session neu an)

*Evidenz:* `find` über das gesamte Repo liefert kein `CLAUDE.md`, `AGENTS.md` oder Äquivalent.
Gleichzeitig ist Agentenbetrieb massiv belegt: 5 `claude/*`-Branches, 3 Worktrees unter
`.claude/worktrees/`, 17 installierte Skills, `skills-lock.json` (Setup vom 2026-06-25 — der
Agentenbetrieb wird gerade hochgefahren).

*Effekt:* Alles, was das README dem Menschen sagt, muss der Agent pro Session neu erschließen — und
was das README *nicht* sagt (Arbeitsbranch ist `develop`, CI gated nur main, `adversarial_keys/`
sind eingefrorene Fixtures), erschließt er womöglich falsch. Das README ist eine gute menschliche
Karte, aber kein Agentengedächtnis.

### F2 — Divergente stale Branches ohne Statusaufzeichnung
**[MED]** · both · Wrong-context-Risk, Verifiability

*Evidenz:* `feat/inverse-node-keys`: 43 Commits vor main, letzter Commit 2025-07-20 (~1 Jahr alt),
existiert auch auf origin. `feat/insert-min`: 1 Commit, 2025-07-25. Nirgendwo (README, Commits,
Doku) ist festgehalten, ob diese Arbeit anhängig oder verworfen ist.

*Effekt:* Aktuelle Wahrheit und Historie sind auf der Branch-Oberfläche nicht unterscheidbar; wer
den 43-Commit-Branch aufnimmt, riskiert Arbeit auf verworfener Grundlage — wer ihn ignoriert,
riskiert Verlust anhängiger Arbeit.

### F3 — Agent-Residuen: leere Branches und verwaiste Worktrees
**[MED]** · agent · Signal-to-noise

*Evidenz:* 6 lokale Branches mit 0 Commits vor main (5× `claude/*`, `worktree-stabilize-workflow`);
3 ausgecheckte Worktrees unter `.claude/worktrees/` (6,2 MB, drei vollständige Repo-Kopien), alle
auf demselben Commit wie main. Die `.gitignore`-Regel für `.claude/` ist erst als **uncommitteter**
Working-Tree-Change vorhanden.

*Effekt:* `git branch` zeigt 9 lokale Branches, von denen 6 keine Arbeit tragen; Werkzeuge, die
`.gitignore` nicht respektieren, sehen jede Quelldatei vierfach. Reines Rauschen auf genau den
Oberflächen (Branch-Liste, Datei-Suche), über die ein Agent navigiert.

### F4 — CI-Gate fehljustiert gegenüber dem Branch-Modell
**[MED]** · both · Verifiability, Wrong-context-Risk

*Evidenz:* `.github/workflows/ci.yml` triggert auf Push zu `[main, clean-up]` — `clean-up` existiert
weder lokal noch auf origin (origin hat nur `main`, `develop`, `feat/inverse-node-keys`). Die
Historie zeigt, dass Arbeit über `develop` läuft (`Merge branch 'develop'` ×2 in den letzten
15 Commits); Pushes auf `develop` lösen kein CI aus.

*Effekt:* Das Validierungs-Gate greift erst beim Merge nach main statt dort, wo die Arbeit passiert —
Fehler werden spät entdeckt (der Commit `fix: resolve CI pipeline failures` direkt vor einem Merge
illustriert genau das). Der tote Trigger-Branch ist zusätzlich eine stale Referenz.

### F5 — Generische Namenskollisionen und gemischte Namenskonvention
**[MED]** · agent · Time-to-correct-context, Signal-to-noise · **skaliert** (jede neue Baumvariante fügt weitere `factory.py`/`base.py` hinzu)

*Evidenz:* Getrackte Duplikate: `factory.py` ×4, `utils.py` ×3, `base.py` ×3, `test_insert.py` ×2.
Im selben Paket koexistieren `src/gplus_trees/g_k_plus/base.py` (abstrakte Basis) und
`src/gplus_trees/g_k_plus/g_k_plus_base.py` (Implementierung), während Geschwisterpakete das Muster
`<name>_base.py` nutzen (`klist_base.py`, `gplus_tree_base.py`, `gk_plus_mkl_base.py`).

*Effekt:* Für IDE-Navigation harmlos (idiomatisches Python-Namespacing — das ist der Trade-off, den
man hier benennen muss), aber für grep/glob-Navigation teuer: „öffne base.py" ist dreideutig, eine
Suche nach `base` streut über sieben Dateien, und die uneinheitliche Konvention macht Dateinamen als
Bedeutungsträger unzuverlässig.

### F6 — Kein Ort für Experiment-Erkenntnisse
**[MED]** · both · Time-to-correct-context · **skaliert** (pro Experiment-Kampagne)

*Evidenz:* Das Repo heißt `gk-tree-experiments`; die Experiment-Outputs sind: ~110 rohe Run-Logs in
`stats/logs/` (1,2 MB, ignoriert, ohne Retention), Benchmark-Ergebnisse außerhalb des Repos
(by design), `BENCHMARK_SUMMARY.md` explizit gitignored. Es existiert kein `docs/`-Layer (Stand vor
diesem Audit), kein Journal, keine ADRs — Schlussfolgerungen („was haben die adversarial-Runs
ergeben?") leben höchstens in Commit-Messages.

*Effekt:* Nach der ~4,5-monatigen Ruhephase (letzter Code-Commit 2026-02-16) ist genau diese
Rekonstruktion der teuerste Teil der Wiederaufnahme — für Mensch wie Agent. Gewichtung hängt von der
Use-Case-Achse ab (siehe Offene Fragen).

### F7 — Duplizierte Skill-Scaffolding, Versionierungsentscheidung offen
**[LOW]** · agent · Wrong-context-Risk

*Evidenz:* `.agents/skills/` und `.claude/skills/` enthalten identische 17 Skills;
`skills-lock.json` (Quelle: `mattpocock/skills`, mit Hashes) verwaltet sie — die Duplikation ist
also *derived*, nicht handgepflegt (das entschärft sie: derived beats hand-maintained). Die
uncommittete `.gitignore`-Änderung („AI agent tooling (lokal, nicht teilen)") würde alle drei
ignorieren.

*Effekt:* Solange die Entscheidung uncommittet ist, ist unklar, welche Kopie autoritativ ist und ob
ein frischer Clone das Agenten-Setup reproduzieren kann. Berührt F1: Wo der künftige
Agent-Einstiegspunkt lebt, hängt an dieser Entscheidung.

### F8 — Streudateien / tote Artefakte
**[LOW]** · both · Signal-to-noise, Wrong-context-Risk

*Evidenz:* (a) `src/__init__.py` ist leer und wird von nichts importiert, macht aber `src` zum
Paket — der `.mypy_cache` enthält dadurch eine zweite Modulidentität `src.gplus_trees.*` parallel
zur installierten `gplus_trees.*`. (b) `tests/logconfig.py` wird von keiner Datei referenziert
(grep über alle `*.py`). (c) Oversized-Testdateien: `tests/gplus/test_insert.py` 1439 Zeilen,
`tests/gk_plus/test_gk_plus_zip.py` 1418, `tests/test_klist.py` 1128 (Source-Maximum dagegen
moderat: 841).

*Effekt:* Geringe, aber unnötige Reibung — die doppelte Modulidentität kann Typechecker-Ergebnisse
verfälschen, die tote Datei kostet einen Fehlgriff, die Monolith-Tests widersetzen sich scoped
Aktivierung.

### F9 — Getrackte Binär-Fixtures mit code-interner Provenienz
**[LOW]** · both · Wrong-context-Risk

*Evidenz:* 24 `.pkl`-Dateien in `benchmarks/adversarial_keys/` sind getrackt; der Generator
`scripts/find_adversarial_keys.py` schreibt nach `adversarial_keys_new/`, die Beförderung in den
getrackten Ordner ist nirgends beschrieben, und die Parameter (counts/K/dim) stehen nur im Skript.

*Effekt:* Als eingefrorene Referenz-Inputs sind getrackte Pickles legitim (Reproduzierbarkeit!) —
der Trade-off kippt nur, wenn ihre Herkunft nicht rekonstruierbar ist. Genau die Provenienz-Kette
ist derzeit implizit.

## 4. Was erhalten werden muss

- **Das README** (`README.md`): aktuell, wahrheitsgetreu (jede stichprobenartig geprüfte Behauptung
  stimmte — Kommandos, Pfade, Skript-Verhalten, Projekt-Layout), mit dokumentierten Flags. Ein
  funktionierender menschlicher Einstiegspunkt — die künftige Agentenschicht sollte darauf zeigen,
  nicht ihn duplizieren.
- **Die Benchmark-Isolation außerhalb des Repos** (`../.isolated-benchmarks/`, Wrapper `benchmark`,
  Doku `benchmarks/BENCHMARKS.md`): Doku und Runner-Code stimmen überein; das Design hält
  Messartefakte komplett aus dem Arbeitsverzeichnis — Signal-to-noise by construction.
- **`.gitignore`-Disziplin**: 103 getrackte Dateien, sauberer `git status`, generierte Artefakte
  (Logs, Caches, `BENCHMARK_SUMMARY.md`) konsequent ignoriert.
- **`pyproject.toml` als Single Source of Truth** für Ruff/Mypy/Pytest — inklusive der ehrlichen,
  datierten Begründung, warum mypy nicht im CI läuft (das ist eine *eingefrorene, autoritative*
  Angabe, kein Drift).
- **Die frische Mixin-Zerlegung** von `g_k_plus` (insert/zip/unzip/navigation/conversion/bulk_create
  als eigene Module) und die **spiegelbildliche Teststruktur** (`tests/gk_plus`, `tests/gplus`,
  `tests/merkle`) — beides senkt Aktivierungskosten und darf ein Aufräumen nicht rückgängig machen.

## 5. Priorisierung (Leverage × Dringlichkeit — was, nicht wie)

1. **F1** — höchster Hebel: Ein Agent-Einstiegspunkt bewegt Time-to-context, Wrong-context-Risk
   *und* Verifiability (indem er das Gate und die Branch-Konvention benennt) gleichzeitig; dringend,
   weil der Agentenbetrieb gerade hochgefahren wird und jede Session ohne ihn die Kosten erneut zahlt.
2. **F4** — Verifiability ist Multiplikator: Ein Gate, das den Arbeitsbranch abdeckt, macht Drift
   überall sonst *detektierbar*.
3. **F2 + F3** — billig und wirksam: Den Status der zwei feat-Branches klären und die 6 leeren
   Branches/3 Worktrees entsorgen bereinigt die Git-Oberfläche, auf der beide Leser navigieren.
4. **F7** — die Versionierungsentscheidung (`.gitignore`-Commit) blockiert die Platzierungsfrage von
   F1; klein, aber vorgelagert.
5. **F6** — Gewichtung hängt von der Use-Case-Antwort ab; wenn Experiment-Auswertung ein Primärtask
   ist, rückt es neben F1.
6. **F5** — nur bei der nächsten neuen Baumvariante relevant (Skalierungsbefund); kein
   Selbstzweck-Umbenennen.
7. **F8 + F9** — Hygiene-Sweep bei Gelegenheit.

## 6. Offene Fragen

1. **Welche Task-Typen dominieren künftig** — Algorithmus-Weiterentwicklung, Experiment-Kampagnen,
   oder Aufschreiben/Publizieren der Ergebnisse? (Entscheidet über das Gewicht von F6 und darüber,
   was in den Agent-Einstiegspunkt gehört.)
2. **Ist `feat/inverse-node-keys` (43 Commits, Juli 2025) anhängige oder verworfene Arbeit** — und
   wo soll dieser Status künftig ablesbar sein?
3. **Soll das Agenten-Scaffolding (`.agents/`, `.claude/skills/`, `skills-lock.json`) geteilt oder
   maschinenlokal sein?** Die uncommittete `.gitignore`-Änderung sagt „lokal" — ist das eine bewusste
   Entscheidung, und wie reproduziert dann ein frischer Clone das Setup?
4. **Ist `develop` weiterhin der Arbeitsbranch**, und soll das Gate ihn abdecken — oder wird auf
   Trunk-based (nur main) umgestellt?
5. **Sind die getrackten adversarial-Key-Pickles eingefrorene Referenz-Inputs** (dann gehört ihre
   Provenienz festgehalten) **oder regenerierbare Caches** (dann stellt sich die Tracking-Frage neu)?

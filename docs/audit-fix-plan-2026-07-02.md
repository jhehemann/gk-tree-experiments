# Umsetzungsplan — Findings F1–F9 (gk-tree-experiments)

**Repo:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments` · **Basis:** `main` @ `bc225d9` · **Remote:** `origin` = `https://github.com/jhehemann/gk-tree-experiments.git`

Reihenfolge nach Audit-Priorisierung: **F1 → F4 → F2+F3 → F7 → F6 → F5 → F8+F9**. Nutzerentscheidungen 1–4 sind eingearbeitet. Alle Pfade absolut. Jeder Schritt hat ein abhakbares Abnahmekriterium.

**Wichtige Reihenfolge-Anmerkung (Konflikt F1↔F7 aufgelöst):** Der Audit priorisiert F1 vor F7, notiert aber selbst, dass die F7-Entscheidung die Platzierungsfrage von F1 „blockiert". Entscheidung 2 löst F7 vorab auf (Scaffolding bleibt maschinenlokal, `CLAUDE.md` im Repo-Root, getrackt). Damit ist die Platzierung von `CLAUDE.md` bereits festgelegt und F1 kann zuerst umgesetzt werden. Der `.gitignore`-Commit (F7) wird jedoch **vor** dem `CLAUDE.md`-Commit ausgeliefert (Commit 1 unten), damit ein sauberer `git status` beim Anlegen von `CLAUDE.md` besteht und keine Scaffolding-Dateien versehentlich mitgetrackt werden.

**Umgebungs-Hinweis (verifiziert, für alle Test-/Install-Schritte relevant):** Es existiert KEIN `./.venv`. Der aktuelle `gk-trees`-Editable-Install liegt (verifiziert via `pip show gk-trees`) in der pyenv-Base 3.11.9 — genau der Zustand, den die globale Nutzerregel verbietet. Die Importierbarkeit von `tests`/`stats` kommt NICHT vom Editable-Install, sondern von der getrackten `conftest.py` (Repo-Root via `sys.path.insert(0, ...)`, Zeilen 8–10). `gplus_trees` wird über das Poetry-Editable (`.pth`) aufgelöst. Build-Backend ist `poetry.core`. Konsequenz für diesen Plan: Test-/Install-Kommandos an ein projektlokales `./.venv` binden oder `poetry install` nennen — nie nacktes `pip install -e .` in die aktive pyenv-Base.

---

## Vorbereitung / Pre-Flight (einmalig, kein Commit)

**P0. Arbeitsbranch anlegen (Konvention aus Entscheidung 3 sofort anwenden).**
- Nicht direkt auf `main` committen. Branch anlegen:
  - `git -C /Users/jannikhehemann/coding/data_structures/gk-tree-experiments switch -c claude/repo-hygiene-2026-07-02`
- Abnahme: `git branch --show-current` → `claude/repo-hygiene-2026-07-02`.
- Begründung: Die in CLAUDE.md festzuhaltende Konvention (Arbeit über `claude/*` + PR gegen `main`) wird hier bereits gelebt; der finale Merge erfolgt per PR.

**P1. Ausgangszustand sichern (Nachvollziehbarkeit).**
- `git -C <repo> branch -vv > /tmp/branches-before.txt` und `git -C <repo> worktree list > /tmp/worktrees-before.txt`.
- Abnahme: beide Dateien existieren und sind nicht leer.

**P2. Baseline grün (damit spätere Änderungen isoliert verifizierbar sind).**
- `git -C <repo> stash` ist NICHT nötig (Working Tree enthält nur `.gitignore`-Mod + untracked `docs/`, beides nicht test-relevant).
- Umgebung projektlokal sicherstellen (kein Install in pyenv-Base):
  - Falls `./.venv` fehlt: `python -m venv /Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.venv` und `<repo>/.venv/bin/pip install -e .` — ODER `poetry install` (Backend ist `poetry.core`).
  - Referenzlauf: `<repo>/.venv/bin/pytest -q` (bzw. `poetry run pytest -q`).
- Abnahme: Testlauf-Exitcode notiert (0 erwartet) sowie Anzahl gesammelter Tests. Wird als Vergleichsbasis für F8-Schritte genutzt.

---

## Commit 1 — F7: Agenten-Scaffolding maschinenlokal fixieren (VORGELAGERT)

*Warum zuerst: Entscheidung 2 legt fest, wo `CLAUDE.md` lebt (Repo-Root, getrackt), und dieser Commit stellt sicher, dass `.claude/`, `.agents/`, `skills-lock.json` nicht versehentlich in spätere Commits geraten. Verifiziert: diese Pfade sind aktuell **untracked** — es ist kein `git rm --cached` nötig, die `.gitignore`-Regel formalisiert nur den Ist-Zustand.*

**Betroffene Datei:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.gitignore` (bereits modifiziert im Working Tree — Diff verifiziert: Block „AI agent tooling (lokal, nicht teilen)" mit `.claude/`, `.agents/`, `skills-lock.json`).

**Schritte:**
1. `.gitignore`-Änderung unverändert übernehmen (Entscheidung 2: nicht anfassen, nur committen).
2. Verifizieren, dass keine Scaffolding-Datei bereits getrackt ist:
   - `git -C <repo> ls-files .claude/ .agents/ skills-lock.json` → muss leer sein (verifiziert leer).
3. Stagen: `git -C <repo> add .gitignore`.
4. Commit: `git -C <repo> commit -m "chore(gitignore): keep AI agent scaffolding machine-local"`.

**Abnahme:**
- `git -C <repo> status --porcelain .gitignore` → leer (committet).
- `git -C <repo> check-ignore .claude .agents skills-lock.json` → listet alle drei.
- `git -C <repo> ls-files .claude/ .agents/` → weiterhin leer (nichts versehentlich getrackt).

**Abhängigkeit:** keine. Muss vor Commit 2 laufen.

---

## Commit 2 — F1: Agent-Einstiegspunkt `CLAUDE.md` (höchster Hebel)

*Warum: Kein Agent-Gedächtnis vorhanden; jede Session leitet Build/Test/Branch-Konventionen neu her. Doktrin aus `documenting-repositories`: always-on-Datei schlank halten, auf README zeigen statt duplizieren, Kontext on demand aktivieren (hub-and-spoke).*

**Neue Datei:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/CLAUDE.md` (Repo-Root, getrackt).

**Inhalt (schlank — nur Invarianten + Zeiger; NICHT das README duplizieren). Stichpunkte:**
- **Kopfzeile / Zweck:** Ein bis zwei Zeilen — „Agent-Einstiegspunkt. Menschliche Doku: README.md. Diese Datei nur Invarianten + Zeiger."
- **Zeiger auf README statt Duplikat:** „Projektüberblick, Installation, Experiment-/Test-Kommandos, Projekt-Layout: siehe `README.md`. Benchmark-Details: `benchmarks/BENCHMARKS.md`."
- **Umgebung & Build/Test/Lint (Invarianten, jede Session gebraucht — kompakt, faktisch korrekt):**
  - **Umgebung:** projektlokales `./.venv` verwenden (Poetry-Backend `poetry.core`). Setup: `python -m venv ./.venv && ./.venv/bin/pip install -e .` ODER `poetry install`. **NIEMALS** `pip install -e .` ohne aktives `./.venv` — landet sonst in der pyenv-Base 3.11.9 (verboten laut globaler Nutzerregel).
  - **Import-Mechanismus (damit Import-Fehler richtig verortet werden):** `tests/` und `stats/` sind importierbar über die getrackte `conftest.py` (Repo-Root via `sys.path.insert`), **nicht** über den editable install; `gplus_trees` kommt aus `src/` über das Poetry-Editable. Bei Import-Problemen zuerst `conftest.py` prüfen.
  - `ruff check .` / `ruff format .` (Lint/Format, Single Source of Truth: `pyproject.toml`).
  - `pytest` (alle Tests) · `pytest -k <name>` · `pytest tests/gplus/`.
  - mypy: `mypy src/` lokal — **läuft bewusst NICHT im CI** (Grund + Re-Enable-Bedingung stehen datiert in `pyproject.toml`, dorthin zeigen, nicht wiederholen).
- **Branch-/PR-Konvention (F4-Entscheidung 3, Trunk-based):**
  - „Trunk-based: `main` ist der einzige Langzeit-Branch. Arbeit auf kurzlebigen `claude/*`-Branches, Integration per Pull Request gegen `main`. Kein `develop` mehr."
  - „CI-Gate greift auf Push zu `main` und auf jeden PR gegen `main`."
- **Ruhende Branches (F2-Entscheidung 1) — als Zeiger, Detail in ADR:**
  - „Zwei Feature-Branches sind ruhend, Entscheidung offen — NICHT darauf aufbauen: `feat/inverse-node-keys`, `feat/insert-min`. Status/Details: `docs/adr/0002-dormant-feature-branches.md`."
- **Agenten-Setup reproduzieren (F7-Konsequenz, Entscheidung 2):**
  - „Skill-Scaffolding (`.claude/`, `.agents/`, `skills-lock.json`) ist maschinenlokal und NICHT im Repo. Reproduktion aus Quelle `mattpocock/skills` via `skills-lock.json`. Ein frischer Clone hat es nicht — das ist gewollt."
- **Wissensablage (F6, Zeiger):**
  - „Experiment-Erkenntnisse: `docs/journal/` (chronologisch). Entscheidungen + Begründung: `docs/adr/`. Fixture-Provenienz: `benchmarks/adversarial_keys/PROVENANCE.md`."
- **Erhaltenswertes (kurzer Hinweis, damit ein Agent es nicht kaputt-„aufräumt"):**
  - „Nicht anfassen ohne Grund: Benchmark-Isolation außerhalb des Repos (`../.isolated-benchmarks/`, Wrapper `./benchmark`), getrackte `.pkl`-Fixtures (eingefrorene Referenz-Inputs), Mixin-Zerlegung in `src/gplus_trees/g_k_plus/`."

**Länge:** eine Bildschirmseite; Detail-Bodies liegen in `docs/` und `pyproject.toml`, hier nur Zeiger (Doktrin: always-on lean).

**Schritte:**
1. Datei `CLAUDE.md` mit obigem Inhalt anlegen.
2. `git -C <repo> add CLAUDE.md`.
3. Commit: `git -C <repo> commit -m "docs(claude): add agent entrypoint (build/test/branch conventions, pointers)"`.

**Abnahme:**
- `CLAUDE.md` existiert im Repo-Root und ist getrackt (`git -C <repo> ls-files CLAUDE.md` nicht leer).
- Die Build-Invariante nennt `./.venv`/`poetry install` (kein nacktes `pip install -e .` in die pyenv-Base) und benennt `conftest.py` als Import-Pfad-Quelle für `tests`/`stats`.
- Jeder Link/Zeiger zeigt auf eine Datei, die nach Abschluss dieses Plans existiert (README.md ✓; `pyproject.toml` ✓; `docs/adr/0002-*.md` und `docs/journal/`, `benchmarks/adversarial_keys/PROVENANCE.md` — werden in Commits 4/5/6 erstellt; falls dieser Commit vor jenen gemergt wird, sind es Vorwärts-Zeiger — akzeptabel, da ein einziger PR den Gesamtstand liefert).
- `CLAUDE.md` enthält KEIN dupliziertes README-Material (Stichprobe: keine kopierten Installations-/Experiment-Absätze).

**Abhängigkeit:** nach Commit 1 (sauberer `.gitignore`-Zustand). Zeiger auf `docs/adr/0002`, `docs/journal/`, `PROVENANCE.md` sind vorwärtsgerichtet; deren Ziele entstehen in Commit 4, 5 und 6 desselben PR.

---

## Commit 3 — F4: CI-Gate auf Trunk-based ausrichten + toten `clean-up`-Trigger entfernen

*Warum: `ci.yml` triggert Push auf `[main, clean-up]` — `clean-up` existiert nicht (verifiziert: origin hat nur `main`, `develop`, `feat/inverse-node-keys`). `pull_request` nur auf `main`. Entscheidung 3: Trunk-based, `develop` wird entfernt (Commit 4), Gate deckt `main` + PRs gegen `main` ab.*

**Betroffene Datei:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.github/workflows/ci.yml` (Zeilen 3–7).

**Änderung (Stichpunkte):**
- `push.branches: [main, clean-up]` → `push.branches: [main]` (toten `clean-up`-Trigger entfernen).
- `pull_request.branches: [main]` bleibt (deckt alle `claude/*`-PRs gegen `main` ab).
- Rest der Datei (Lint-Job, Test-Matrix 3.10/3.11/3.12) unverändert.

**Schritte:**
1. In `ci.yml` `branches: [main, clean-up]` durch `branches: [main]` ersetzen.
2. YAML-Validität prüfen: `python -c "import yaml,sys; yaml.safe_load(open('/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.github/workflows/ci.yml'))"` (falls PyYAML fehlt: `yamllint` oder rein visuelle Prüfung — mindestens Einrückung/Struktur bestätigen).
3. `git -C <repo> add .github/workflows/ci.yml`.
4. Commit: `git -C <repo> commit -m "ci: gate on main only, drop dead clean-up trigger (trunk-based)"`.

**Abnahme:**
- `git -C <repo> diff HEAD~1 -- .github/workflows/ci.yml` zeigt genau die Trigger-Zeile geändert, sonst nichts.
- `grep -n "clean-up" <repo>/.github/workflows/ci.yml` → kein Treffer.
- YAML lädt fehlerfrei.

**Abhängigkeit:** logisch vor Commit 4 (Trigger darf `develop` nicht mehr referenzieren — tut er ohnehin nicht; Reihenfolge ist Sicherheits-, keine harte Abhängigkeit).

---

## Commit 4 — F2 + F3: Git-Oberfläche bereinigen (Branches, Worktrees, ruhende Feature-Branches dokumentieren)

*Warum: 6 leere lokale Branches + `develop` (alle verifiziert 0 Commits vor `main`) und 3 verwaiste Worktrees (alle auf `bc225d9`) sind Rauschen. Die zwei `feat/*`-Branches bleiben und werden nur dokumentiert (Entscheidung 1).*

**WICHTIG — Randbedingungen strikt:** Worktrees nur mit `git worktree remove` (nicht `rm -rf`). Branch-Löschungen nur für die 6 leeren + `develop`. `feat/inverse-node-keys` und `feat/insert-min` bleiben unangetastet.

### 4a. Verwaiste Worktrees entfernen (3 Stück, alle @ `bc225d9`)
- `git -C <repo> worktree remove /Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.claude/worktrees/charming-chaplygin-6b38d9`
- `git -C <repo> worktree remove /Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.claude/worktrees/goofy-beaver-d2132d`
- `git -C <repo> worktree remove /Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.claude/worktrees/stabilize-workflow`
- Danach `git -C <repo> worktree prune`.
- Falls `worktree remove` wegen „nicht sauber" abbricht (sollte nicht — alle auf main-Commit, keine Änderungen erwartet): Zustand des betreffenden Worktrees mit `git -C <worktree-pfad> status` prüfen, KEIN `--force` ohne erneute Bestätigung, dass 0 Commits/keine ungespeicherte Arbeit vorliegen.

### 4b. Leere lokale Branches löschen (6 Stück — nach 4a, da 3 davon Worktree-Branches waren)
- Zuerst die drei nun freigegebenen Worktree-Branches:
  - `git -C <repo> branch -d claude/charming-chaplygin-6b38d9`
  - `git -C <repo> branch -d claude/goofy-beaver-d2132d`
  - `git -C <repo> branch -d worktree-stabilize-workflow`
- Dann die drei reinen leeren Branches:
  - `git -C <repo> branch -d claude/blissful-agnesi-456fbc`
  - `git -C <repo> branch -d claude/elegant-engelbart-57bb1c`
  - `git -C <repo> branch -d claude/quirky-cartwright-042a56`
- `-d` (nicht `-D`): scheitert bewusst, falls ein Branch doch nicht vollständig in `main` ist — Sicherheitsnetz. Da alle verifiziert 0 vor main, muss `-d` durchlaufen.

### 4c. `develop` löschen — lokal und auf origin (Entscheidung 3, verifiziert 0 vor / 5 hinter main)
- **Just-in-time-Recheck VOR dem Remote-Delete (Pflicht, da einzige irreversible/netzabhängige Operation):**
  - `git -C <repo> fetch origin`
  - `git -C <repo> rev-list --left-right --count origin/main...origin/develop` — die rechte Zahl (develop ahead) MUSS `0` sein (aktuell verifiziert: `5  0`). Ist sie ≠ 0, hat sich `origin/develop` seit der Planung bewegt → **4c ABBRECHEN, KEIN Delete**, Zustand an den Nutzer melden.
- **Tip-SHA dauerhaft festhalten (Rollback-Absicherung):** `develop`-Tip ist `df17bb21ac86a683672e4013fdfde25ed7865303`. Dieser SHA + Wiederherstellungs-Einzeiler gehören in den Commit-Body von Commit 4 UND in ADR-0002 (4d):
  - Wiederherstellung bei Bedarf: `git push origin df17bb21ac86a683672e4013fdfde25ed7865303:refs/heads/develop`
- Erst nach grünem Recheck löschen:
  - Lokal: `git -C <repo> branch -d develop`.
  - Origin: `git -C <repo> push origin --delete develop`.
- Begründung: Trunk-based; `develop` trägt keine eigene Arbeit (0 Commits vor main), nichts geht verloren.

### 4d. Ruhende Feature-Branches dokumentieren (F2, Entscheidung 1 — NICHTS löschen)
**Neue Datei:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/adr/0002-dormant-feature-branches.md` (ADR-Form: append-only Entscheidungsprotokoll; Nummerierung siehe Commit 5, das ADR-0001 anlegt).
- Inhalt (Stichpunkte):
  - **Status:** „Ruhend — Entscheidung offen. NICHT darauf aufbauen ohne vorherige Klärung."
  - **Datum:** 2026-07-02.
  - `feat/inverse-node-keys`: 43 Commits vor `main`, 112 hinter, letzter Commit 2025-07-20 (~1 Jahr alt), existiert auf origin. Zweck: siehe Branch-Historie / Namensgebung (inverse Node-Keys). Unklar, ob anhängig oder verworfen.
  - `feat/insert-min`: 1 WIP-Commit (`6f26237`), 2025-07-25, nur lokal. Commit-Message dokumentiert bekanntes Problem: „not working with add_dummy flag" — funktioniert derzeit nicht vollständig.
  - **Warum nicht gelöscht:** Status unklar; Löschen riskiert Verlust anhängiger Arbeit. Aufbewahrt bis Entscheidung getroffen.
  - **Entfernter `develop`-Branch (Rollback-Notiz):** `develop` wurde im Zuge der Trunk-based-Umstellung entfernt (lokal + origin), verifiziert 0 Commits vor `main`. Tip-SHA `df17bb21ac86a683672e4013fdfde25ed7865303` festgehalten; Wiederherstellung: `git push origin df17bb21ac86a683672e4013fdfde25ed7865303:refs/heads/develop`.
  - **Nächster Schritt / wer entscheidet:** offen — bei Wiederaufnahme des jeweiligen Themas Rebase-Kosten (112 Commits Divergenz bei `inverse-node-keys`) gegen Nutzen abwägen.
- Begründung ADR statt README-Eintrag: Es ist eine Entscheidung mit Begründung (Doktrin: „Decision + rationale → ADR, append-only, getrennt von current truth"). `CLAUDE.md` zeigt nur darauf.

**Schritte (Commit):**
1. 4a–4c ausführen (Git-Operationen, kein Datei-Commit), inkl. JIT-Recheck vor 4c.
2. `docs/adr/`-Verzeichnis anlegen falls noch nicht durch Commit 5 existent; `0002-dormant-feature-branches.md` schreiben.
3. `git -C <repo> add docs/adr/0002-dormant-feature-branches.md`.
4. Commit: `git -C <repo> commit -m "chore(git): prune empty branches + orphan worktrees; document dormant feat branches"` — Commit-Body enthält den `develop`-Tip-SHA `df17bb2…` + Wiederherstellungs-Einzeiler.

**Abnahme:**
- JIT-Recheck-Nachweis: `git rev-list --left-right --count origin/main...origin/develop` wurde unmittelbar vor 4c ausgeführt und zeigte rechte Zahl `0` (sonst Abbruch dokumentiert).
- `git -C <repo> worktree list` → nur noch der Haupt-Worktree.
- `git -C <repo> branch` → nur `main`, `feat/inverse-node-keys`, `feat/insert-min`, `claude/repo-hygiene-2026-07-02` (aktueller Arbeitsbranch). Kein `develop`, keine leeren `claude/*`, kein `worktree-stabilize-workflow`.
- `git -C <repo> branch -r` → `origin/develop` nicht mehr vorhanden; `origin/main`, `origin/feat/inverse-node-keys` bleiben.
- `feat/inverse-node-keys` und `feat/insert-min` unverändert vorhanden (`git -C <repo> rev-parse feat/inverse-node-keys` = `9094930…`, `feat/insert-min` = `6f26237…`).
- `docs/adr/0002-dormant-feature-branches.md` existiert, nennt beide Branches mit Status „ruhend" UND den `develop`-Tip-SHA samt Wiederherstellungs-Einzeiler.
- Verzeichnis `.claude/worktrees/` ist leer/entfernt (ignoriert, daher kein `git status`-Effekt).

**Abhängigkeit:** 4a vor 4b (Worktree-Branches erst nach Worktree-Entfernung löschbar). Netzzugriff für 4c (`fetch` + `push origin --delete`). `docs/adr/` sollte durch Commit 5 (ADR-0001) etabliert sein — falls Commit 4 vor Commit 5 läuft, Verzeichnis hier miterstellen.

---

## Commit 5 — F6: Wissens-Layer `docs/` etablieren (Journal + ADR-Rahmen)

*Warum: Entscheidung 4 gibt F6 volles Gewicht (alle drei Task-Typen dominieren). Kein Ort für Experiment-Erkenntnisse; Run-Logs ohne Retention; `BENCHMARK_SUMMARY.md` gitignored. Doktrin: Journal = evolving history/handoff; ADR = Entscheidung+Begründung; beide on-demand, aus `CLAUDE.md` verlinkt (hub-and-spoke).*

**Anmerkung Reihenfolge:** Das Audit reiht F6 nach F2+F3/F7. `docs/` wird hier als eigener Commit etabliert; das bereits existierende `docs/repository-audit-2026-07-02.md` (untracked) wird mit-eingecheckt.

**Neue/betroffene Dateien:**
- `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/README.md` — Index/Wegweiser des docs-Layers.
- `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/journal/README.md` — Zweck + Format des Experiment-Journals.
- `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/adr/0001-record-decisions-in-adrs.md` — ADR-0001 (etabliert ADR-Prozess selbst).
- `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/adr/README.md` — ADR-Index + Nummerierungs-/Format-Konvention.
- `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/docs/repository-audit-2026-07-02.md` — bereits vorhanden (untracked), jetzt einchecken.

**Inhalte (Stichpunkte):**
- `docs/README.md`: „Wissensschicht. `journal/` = chronologische Experiment-Erkenntnisse; `adr/` = Entscheidungen+Begründung (append-only); `repository-audit-*.md` = Punkt-in-Zeit-Audits. Für Agenten-Invarianten siehe Repo-Root `CLAUDE.md`, für menschlichen Einstieg `README.md`."
- `docs/journal/README.md`:
  - Zweck: „Was haben die Experiment-Runs ergeben — Erkenntnisse, die sonst nur in Commit-Messages/verworfenen Logs leben."
  - Format: eine Datei pro Eintrag, Namensschema `YYYY-MM-DD-<slug>.md`; Kopf mit Datum, Kontext (welche Kampagne/Parameter), Beobachtung, Schlussfolgerung, Verweise (Commit-Hashes, ggf. `../.isolated-benchmarks/`-Läufe).
  - Retention-Hinweis: „Run-Logs unter `stats/logs/` sind flüchtig (ignoriert, keine Retention) — dauerhafte Erkenntnis gehört hierher, nicht in die Logs."
  - Optional ein Platzhalter-/Beispiel-Eintrag NUR wenn eine reale Erkenntnis vorliegt; sonst keine Spekulativ-Datei (Doktrin: nicht auf Vorrat schreiben). Standard: nur `README.md` im Journal.
- `docs/adr/0001-record-decisions-in-adrs.md`: klassisches ADR-0001 („We will use ADRs"), Status Accepted, Datum 2026-07-02; definiert Format (Titel, Status, Datum, Kontext, Entscheidung, Konsequenzen), Nummerierung fortlaufend, append-only, Änderung nur durch Supersede-ADR.
- `docs/adr/README.md`: Liste der ADRs (0001, 0002) + „Neuen ADR anlegen: nächste freie Nummer, obiges Format."

**Schritte:**
1. Verzeichnisse `docs/journal/` und `docs/adr/` anlegen (falls `docs/adr/` schon durch Commit 4 existiert, nur ergänzen).
2. Obige Dateien schreiben.
3. Stagen: `git -C <repo> add docs/` (nimmt auch das bereits vorhandene Audit + ADR-0002 aus Commit 4 auf, falls noch nicht committet — hier prüfen, dass ADR-0002 bereits in Commit 4 committet wurde, sonst separat behandeln).
4. Commit: `git -C <repo> commit -m "docs: establish knowledge layer (journal + ADR framework, check in audit)"`.

**Abnahme:**
- `docs/README.md`, `docs/journal/README.md`, `docs/adr/0001-record-decisions-in-adrs.md`, `docs/adr/README.md` existieren und sind getrackt.
- `docs/repository-audit-2026-07-02.md` ist jetzt getrackt (`git -C <repo> ls-files docs/repository-audit-2026-07-02.md` nicht leer).
- Alle Zeiger aus `CLAUDE.md` (Commit 2) auf `docs/journal/`, `docs/adr/` sind jetzt eingelöst (keine toten Links im finalen PR-Stand).
- `git -C <repo> status --porcelain docs/` → leer (alles committet).

**Abhängigkeit:** ADR-0001 sollte vor/mit ADR-0002 existieren (Nummernkonsistenz). Falls Commit 4 zuerst lief und ADR-0002 anlegte, ist das ok — beide landen im selben PR; sicherstellen, dass `docs/adr/README.md` beide listet.

---

## Commit 6 — F9: Fixture-Provenienz dokumentieren

*Warum: 23 getrackte `.pkl` (verifiziert — Audit sagt 24, tatsächlich 23) in `benchmarks/adversarial_keys/`; Generator `scripts/find_adversarial_keys.py` erzeugt ein `adversarial_keys_new/`-Verzeichnis on-demand (verifiziert: existiert derzeit NICHT im Repo; wird zur Laufzeit via `os.makedirs(exist_ok=True)` an `scripts/find_adversarial_keys.py:71` angelegt); Beförderungsprozess + Parameter nirgends dokumentiert. Entscheidung 4: Provenienz gehört in den Plan. Erhaltenswert: die Pickles bleiben getrackt (eingefrorene Referenz-Inputs).*

**Neue Datei:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/benchmarks/adversarial_keys/PROVENANCE.md`
**Zusätzlich betroffen:** `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/.gitignore` (neue Zeile `benchmarks/adversarial_keys_new/`, siehe unten).

**Inhalt PROVENANCE.md (Stichpunkte):**
- **Was:** 23 eingefrorene `.pkl`-Fixtures mit adversarialen rank-1-Keys; bewusst getrackt für Reproduzierbarkeit von Benchmarks.
- **Namensschema:** `keys_sz<size>_k<K>_d<dim>.pkl` (aus Dateiliste ableiten, z. B. `keys_sz10000_k16_d160.pkl` = size 10000, K=16, dim 160). Vorhandene Kombinationen tabellarisch: sizes = {10000}, K ∈ {4,8,16,32}, dim-Werte je K (aus Dateiliste: K4 → d1/10/20/30/40; K8 → d1/10/20/40/80; K16 → d1/10/20/40/80/160; K32 → d1/10/20/40/80/160/320).
- **Woher / Generator:** `scripts/find_adversarial_keys.py` erzeugt Keys in ein on-demand angelegtes Verzeichnis `benchmarks/adversarial_keys_new/` (Pfad-Definition + `os.makedirs(exist_ok=True)` an `scripts/find_adversarial_keys.py:70–71`); Parameter (counts/K/dim) sind im Skript nahe dem Aufruf/`__main__` definiert — Verweis auf konkrete Codezeile/Funktion (beim Schreiben im Skript nachschlagen und exakte Stelle nennen, nicht raten).
- **Beförderungsprozess (der bisher implizite Schritt, jetzt explizit):** „Neue Keys landen in `adversarial_keys_new/` (existiert erst, wenn der Generator läuft). Nach Sichtung/Validierung werden ausgewählte Dateien manuell nach `adversarial_keys/` verschoben und committet — nur dann werden sie zu eingefrorenen Referenz-Inputs. `adversarial_keys_new/` ist per `.gitignore` ausgeschlossen (in diesem PR ergänzt) und darf nicht committet werden."
- **Verweis:** README-Zeile 81 (Generator schreibt nach `adversarial_keys_new/`) bleibt Single Source für das Generator-Verhalten; PROVENANCE ergänzt nur die Beförderungs-/Herkunftskette.

**.gitignore-Änderung (macht die „nicht getrackt"-Zusicherung erst wahr):**
- Zeile `benchmarks/adversarial_keys_new/` hinzufügen. Ohne diese Regel würden vom Generator erzeugte `.pkl` als untracked auftauchen und könnten versehentlich committet werden — genau die Disziplin, die F9 schützen will.

**Schritte:**
1. `scripts/find_adversarial_keys.py` kurz öffnen, exakte Parameter-Fundstelle (Zeile/Funktion) und Default-counts notieren, in PROVENANCE eintragen (keine Annahme — verifizieren). Verzeichnis-Pfad + `makedirs` an Zeile 70–71 verifiziert.
2. `.gitignore` um `benchmarks/adversarial_keys_new/` ergänzen.
3. `PROVENANCE.md` schreiben.
4. `git -C <repo> add benchmarks/adversarial_keys/PROVENANCE.md .gitignore`.
5. Commit: `git -C <repo> commit -m "docs(benchmarks): document adversarial-key fixture provenance and promotion; ignore adversarial_keys_new"`.

**Abnahme:**
- `benchmarks/adversarial_keys/PROVENANCE.md` existiert, getrackt.
- Parameter-Verweis zeigt auf eine reale Zeile in `scripts/find_adversarial_keys.py` (nachgeprüft, nicht geraten).
- `git -C <repo> check-ignore benchmarks/adversarial_keys_new/` → Treffer (Regel greift).
- Die 23 `.pkl` bleiben getrackt und unverändert (`git -C <repo> ls-files benchmarks/adversarial_keys/*.pkl | wc -l` → 23).
- Zeiger aus `CLAUDE.md` auf `PROVENANCE.md` ist eingelöst.

**Abhängigkeit:** keine harte; sinnvoll nach Commit 2 (CLAUDE.md verweist darauf).

---

## Commit 7 — F8: Streudateien / tote Artefakte bereinigen

*Warum: Geringe, aber unnötige Reibung. Drei Teilbefunde, getrennt behandelt. Erhaltenswert: Mixin-Zerlegung und Teststruktur dürfen nicht beschädigt werden. Baseline-Tests aus P2 als Kontrollpunkt.*

### 7a. Leeres `src/__init__.py` entfernen (doppelte Modulidentität `src.gplus_trees.*`)
- Datei: `/Users/jannikhehemann/coding/data_structures/gk-tree-experiments/src/__init__.py` (0 Zeilen, verifiziert leer, getrackt).
- **Vor Löschen prüfen**, dass nichts `import src…` oder `from src…` nutzt:
  - `grep -rn -E "(^|[^.])\bsrc\.gplus_trees|from src\b|import src\b" <repo> --include="*.py"` → erwartet keine Treffer (leer). Falls Treffer: NICHT löschen, stattdessen im PR-Text vermerken und diesen Teilschritt zurückstellen.
  - Kontext (kein setuptools-„src-layout" hier): Poetry paketiert via `packages = [{ include = "gplus_trees", from = "src" }]` — `src` selbst ist KEIN Paket, sondern nur durch das leere `__init__.py` + `conftest.py`-sys.path-insert versehentlich importierbar (genau die doppelte Modulidentität aus F8). Der grep-Nachweis (kein `import src`) genügt für die sichere Löschung; `gplus_trees` wird über die `.pth`-Datei unabhängig von `src/__init__.py` aufgelöst. Kein „src-layout"-Jargon anwenden.
- Löschen: `git -C <repo> rm src/__init__.py`.

### 7b. Tote `tests/logconfig.py` entfernen + dangling commented Imports säubern
- **Korrektur zum Audit:** `tests/logconfig.py` ist NICHT völlig unreferenziert — es wird in zwei Dateien nur als **auskommentierter** Import erwähnt (verifiziert):
  - `tests/gk_plus/test_gk_plus_tree.py:25` → `# from tests.logconfig import logger`
  - `tests/gk_plus/test_split_inplace.py:20` → `# from tests.logconfig import logger`
- Runtime ist die Datei tot (kein aktiver Import). Schritte:
  - `git -C <repo> rm tests/logconfig.py`.
  - Die zwei auskommentierten Import-Zeilen aus beiden Testdateien entfernen (sonst zeigen sie auf eine nicht mehr existierende Datei → irreführend für Agent/Mensch).
- **Wenn** einer der beiden Kommentare bewusst als „so würde man Logging aktivieren"-Hinweis dienen soll: dann Datei behalten und stattdessen einen Ein-Zeilen-Hinweis ergänzen. Standard-Annahme: entfernen (siehe Assumptions).

### 7c. Monolith-Testdateien — NICHT in diesem Sweep aufteilen
- Betroffen: `tests/gplus/test_insert.py` (1439 Z.), `tests/gk_plus/test_gk_plus_zip.py` (1418 Z.), `tests/test_klist.py` (1128 Z.).
- **Bewusst KEIN Split in diesem Plan:** Test-Splitting ist inhaltliche Arbeit mit Regressionsrisiko, gehört nicht in einen Hygiene-Sweep und ist im Audit als „Signal-to-noise, LOW" eingestuft. Stattdessen: Als Journal-/Backlog-Notiz festhalten.
  - Ein-Zeilen-Eintrag in `docs/journal/README.md` unter „Offener Aufräum-Backlog" ODER Verweis in ADR — Standard: kurze Backlog-Zeile in `docs/README.md` bzw. neuer `docs/BACKLOG.md` (leichtgewichtig).
  - Alternativ als separater spawn_task/Issue außerhalb dieses Plans. (Siehe Assumption 6.)

**Schritte (Commit):**
1. 7a-Guards ausführen; bei grünem Guard `src/__init__.py` löschen.
2. `tests/logconfig.py` löschen; zwei kommentierte Import-Zeilen entfernen.
3. Regressionslauf: `<repo>/.venv/bin/pytest -q` (bzw. `poetry run pytest -q`) — muss identischen Status wie P2-Baseline liefern (Exitcode + Anzahl gesammelter Tests). Kein Install in die pyenv-Base.
4. Optional: `.mypy_cache` lokal löschen, damit die alte `src.gplus_trees.*`-Identität verschwindet (`rm -rf <repo>/.mypy_cache`); danach `mypy src/` — kein neuer Fehler ggü. vorher (mypy ist ignoriert im CI, daher nicht blockierend, nur lokale Bestätigung).
5. Backlog-Notiz für 7c schreiben.
6. `git -C <repo> add -A` (erfasst Löschungen + Editierungen + Backlog-Datei).
7. Commit: `git -C <repo> commit -m "chore: remove dead src/__init__.py and tests/logconfig.py; note oversized-test backlog"`.

**Abnahme:**
- `src/__init__.py` und `tests/logconfig.py` sind gelöscht (`git -C <repo> ls-files | grep -E "src/__init__.py|tests/logconfig.py"` → leer).
- `grep -rn "logconfig" <repo> --include="*.py"` → kein Treffer mehr (auch keine Kommentare).
- `pytest -q` grün, Anzahl gesammelter Tests unverändert ggü. P2 (kein versehentlich verlorener Test).
- Mixin-Module unter `src/gplus_trees/g_k_plus/` unverändert (Erhaltenswertes intakt).
- 7c: Backlog-Zeile für die drei Monolith-Tests dokumentiert; Testdateien selbst unverändert.

**Abhängigkeit:** P2-Baseline muss vorliegen (Vergleich). Kein Netzzugriff.

---

## Abschluss — PR gegen `main` (Trunk-based-Konvention leben)

**A1. Branch pushen und PR öffnen.**
- `git -C <repo> push -u origin claude/repo-hygiene-2026-07-02`.
- PR gegen `main` erstellen (gh CLI falls vorhanden: `gh pr create --base main --fill`), Titel z. B. „Repo hygiene: agent entrypoint, CI gate, git cleanup, docs layer (F1–F9)".
- PR-Beschreibung: Findings F1–F9 mit Häkchen, Verweis auf `docs/repository-audit-2026-07-02.md`, explizit notiert: `feat/*`-Branches bewusst erhalten (Entscheidung 1), F5 bewusst nicht umgesetzt (siehe unten), F8c (Test-Split) als Backlog, `develop`-Tip-SHA `df17bb2…` + Wiederherstellungsbefehl (Rollback).
- Abnahme: PR existiert, Base = `main`, CI (neuer Trigger) läuft auf dem PR grün.

**A2. Merge.**
- Nach grünem CI mergen; `claude/repo-hygiene-2026-07-02` nach Merge löschen (lokal `-d`, remote `--delete`).
- Abnahme: `main` enthält alle Commits 1–7; Arbeitsbranch entsorgt.

---

## F5 — bewusst NICHT umgesetzt (dokumentierte Nicht-Entscheidung)

*Audit + Randbedingung: F5 („generische Namenskollisionen", `factory.py`×4, `base.py`×3, `utils.py`×3, `test_insert.py`×2; `g_k_plus/base.py` vs. `g_k_plus/g_k_plus_base.py`) ist ein Skalierungsbefund — nur bei der nächsten neuen Baumvariante relevant, kein Selbstzweck-Umbenennen. Idiomatisches Python-Namespacing; für IDE-Navigation harmlos.*

- **Aktion jetzt:** KEIN Umbenennen. Stattdessen eine kurze Konventions-Notiz für künftige Varianten festhalten, damit die Inkonsistenz nicht weiterwächst:
  - Eintrag in `docs/adr/` (z. B. ADR-0003 „Naming convention for tree-variant packages") ODER eine Zeile in `docs/README.md`/`BACKLOG.md`: „Bei neuer Baumvariante: `<name>_base.py` für die Implementierungsbasis verwenden (konsistent zu `klist_base.py`, `gplus_tree_base.py`, `gk_plus_mkl_base.py`); `base.py` bleibt der abstrakten Basis vorbehalten. Bestehendes `g_k_plus/base.py` vs. `g_k_plus/g_k_plus_base.py` erst beim nächsten Anfassen des Pakets angleichen."
- Diese Notiz kann als Teil von Commit 5 (docs-Layer) mitgeliefert werden — sie erzeugt keinen Code-Change und kein Regressionsrisiko.
- **Abnahme:** Konvention ist an genau einer Stelle in `docs/` festgehalten; kein `git mv` in diesem PR.

---

## Commit-Schnitt-Übersicht (Reihenfolge = Umsetzungsreihenfolge)

| # | Finding(s) | Commit-Message (Vorschlag) | Kern-Artefakte |
|---|-----------|----------------------------|----------------|
| 1 | F7 | `chore(gitignore): keep AI agent scaffolding machine-local` | `.gitignore` |
| 2 | F1 | `docs(claude): add agent entrypoint (build/test/branch conventions, pointers)` | `CLAUDE.md` |
| 3 | F4 | `ci: gate on main only, drop dead clean-up trigger (trunk-based)` | `.github/workflows/ci.yml` |
| 4 | F2+F3 | `chore(git): prune empty branches + orphan worktrees; document dormant feat branches` | Git-Ops (inkl. JIT-Recheck + develop-Rollback-Notiz) + `docs/adr/0002-*.md` |
| 5 | F6 (+F5-Notiz) | `docs: establish knowledge layer (journal + ADR framework, check in audit)` | `docs/README.md`, `docs/journal/`, `docs/adr/0001`, `docs/adr/README.md`, Audit einchecken |
| 6 | F9 | `docs(benchmarks): document adversarial-key fixture provenance and promotion; ignore adversarial_keys_new` | `benchmarks/adversarial_keys/PROVENANCE.md`, `.gitignore` |
| 7 | F8 | `chore: remove dead src/__init__.py and tests/logconfig.py; note oversized-test backlog` | Löschungen + Backlog-Notiz |

*Alle 7 Commits landen in einem PR (`claude/repo-hygiene-2026-07-02` → `main`); Vorwärts-Zeiger in `CLAUDE.md` sind im finalen PR-Stand vollständig eingelöst. Wer feiner schneiden will, kann Commits 1–4 (Substrat) und 5–7 (Wissens-/Hygiene-Layer) auf zwei PRs aufteilen — dann müssen die `docs/`-Zeiger aus `CLAUDE.md` in PR 1 als „folgt in PR 2" markiert oder auskommentiert werden.*

---

## Erhaltenswertes — Regressions-Checkliste (nach allen Commits)

- [ ] `README.md` unverändert (nur ggf. bewusst; dieser Plan ändert es NICHT).
- [ ] `../.isolated-benchmarks/`, `./benchmark`, `benchmarks/BENCHMARKS.md` unberührt.
- [ ] `pyproject.toml` unverändert (Single Source of Truth; mypy-Begründung intakt) — außer optionaler F5-Notiz, die aber in `docs/` liegt, nicht hier.
- [ ] Mixin-Zerlegung `src/gplus_trees/g_k_plus/*` und Teststruktur `tests/{gk_plus,gplus,merkle}` unverändert.
- [ ] 23 `.pkl`-Fixtures weiterhin getrackt.
- [ ] `pytest` grün (via `./.venv`/`poetry`, nicht pyenv-Base), Testanzahl == P2-Baseline.
- [ ] `git status` sauber (keine ungewollten untracked/modified Reste).

---

## Annahmen (explizit, vom Plan referenziert)

1. Netzzugriff auf origin (GitHub) ist verfügbar — benötigt für Commit 4c (`git push origin --delete develop`) und den Abschluss-PR. Ohne Netz laufen alle lokalen Schritte (Commits 1–3, lokaler Teil von 4, 5–7); die Origin-Löschung und der PR verschieben sich.
2. `git -C <repo> switch -c claude/repo-hygiene-2026-07-02` (Pre-Flight P0) ist gewünscht — der Plan lebt die in CLAUDE.md festgehaltene Trunk-based-Konvention (Arbeit über claude/*-Branch + PR gegen main) sofort. Falls stattdessen direkt auf main gearbeitet werden soll, entfällt P0/A1/A2 und die Commits gehen direkt auf main.
3. Die uncommittete `.gitignore`-Änderung wird exakt wie im aktuellen Working Tree übernommen (Entscheidung 2) — kein zusätzliches `git rm --cached`, da `.claude/`, `.agents/`, `skills-lock.json` verifiziert bereits untracked sind.
4. F8b: Die zwei auskommentierten `# from tests.logconfig import logger`-Zeilen in `tests/gk_plus/test_gk_plus_tree.py:25` und `tests/gk_plus/test_split_inplace.py:20` sind Alt-Rückstände und werden mit der Datei entfernt. Falls sie bewusst als Aktivierungshinweis für Logging dienen sollen, bleibt `tests/logconfig.py` erhalten und bekommt stattdessen einen Ein-Zeilen-Kommentar (Abweichung im Schritt notiert).
5. Das Audit nennt 24 getrackte `.pkl`-Fixtures; verifiziert sind es 23 (`git ls-files benchmarks/adversarial_keys/ | wc -l` = 23). Der Plan (F9) arbeitet mit der verifizierten Zahl 23. `benchmarks/adversarial_keys_new/` ist leer/untracked.
6. F8c (Aufteilung der drei Monolith-Testdateien) und F5 (Umbenennungen) werden in diesem Plan NICHT ausgeführt, sondern nur als Backlog/Konventions-Notiz in `docs/` festgehalten — konform zur Audit-Priorisierung (LOW/Skalierungsbefunde, Regressionsrisiko in einem Hygiene-Sweep unerwünscht). Bei Wunsch nach sofortiger Umsetzung wären das eigene, test-getriebene PRs.
7. `src/`-Layout ist ein src-layout (Paket-Root), in dem `src` selbst kein installierbares Paket ist; das leere `src/__init__.py` ist daher gefahrlos entfernbar, sobald der grep-Guard in Schritt 7a keine `import src…`-Nutzung findet. Falls der Guard Treffer liefert, wird 7a zurückgestellt.
8. ADR-Nummerierung: 0001 etabliert den ADR-Prozess selbst, 0002 dokumentiert die ruhenden Feature-Branches; eine optionale 0003 könnte die Namenskonvention (F5) aufnehmen. Reihenfolge der Commits 4/5 muss nur sicherstellen, dass `docs/adr/README.md` am Ende beide/alle ADRs listet.
9. README.md wird durch diesen Plan nicht verändert (Erhaltenswert); insbesondere bleibt Zeile 81 die Single Source für das Generator-Verhalten, PROVENANCE.md ergänzt nur die Beförderungs-/Herkunftskette.

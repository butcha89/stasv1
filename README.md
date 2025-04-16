# Stash API Tools

Eine Sammlung von Tools zur Analyse und Verwaltung von Stash-Daten mit Fokus auf Performer-Statistiken, Empfehlungen und Discord-Integration.

## Funktionen

- **Statistik-Modul**: Analysiert Cup-Größen, O-Counter und verschiedene Verhältnisse
- **Empfehlungs-Modul**: Schlägt ähnliche Performer und Szenen basierend auf Ihren Vorlieben vor
- **Dashboard-Modul**: Interaktives Web-Dashboard zur Visualisierung von Statistiken und Empfehlungen
- **Updater-Modul**: Aktualisiert Performer-Daten mit Cup-Größen und Verhältnis-Informationen
- **Discord-Modul**: Sendet Statistiken und Empfehlungen an Discord über Webhooks

## Installation

### Voraussetzungen

- Python 3.7 oder höher
- Stash-Server mit GraphQL API
- Pip (Python-Paketmanager)

### Abhängigkeiten installieren

```bash
pip install requests pandas numpy matplotlib seaborn scikit-learn dash plotly
```

### Konfiguration

1. Klonen Sie das Repository oder laden Sie die Dateien herunter
2. Führen Sie das Skript einmal aus, um die Standard-Konfigurationsdatei zu erstellen:

```bash
python main.py
```

3. Bearbeiten Sie die Datei `config/configuration.ini` und geben Sie Ihre Stash-API-Informationen und Discord-Webhook-URL ein

## Verwendung

### Statistiken generieren

```bash
python main.py stats
```

Speichern der Statistiken in eine Datei:

```bash
python main.py stats --output stats.json
```

### Empfehlungen generieren

```bash
python main.py recommend --type performers
python main.py recommend --type scenes
python main.py recommend --type all
```

### Dashboard starten

```bash
python main.py dashboard --port 8050
```

Öffnen Sie dann einen Browser und navigieren Sie zu `http://localhost:8050`

### Performer-Daten aktualisieren

```bash
python main.py update --type cup-sizes
python main.py update --type ratios
python main.py update --type all
```

### Updates an Discord senden

```bash
python main.py discord --type stats
python main.py discord --type performers
python main.py discord --type scenes
python main.py discord --type all
```

## Module

### Statistik-Modul

Analysiert Ihre Stash-Daten und generiert Statistiken zu:
- Cup-Größen-Verteilung
- O-Counter nach Cup-Größe
- Verhältnisse wie Cup-to-BMI, Cup-to-Height, Cup-to-Weight

### Empfehlungs-Modul

- **Performer-Empfehlungen**: Findet ähnliche Performer basierend auf Cup-Größe, Körpermaßen und anderen Faktoren
- **Szenen-Empfehlungen**: Schlägt Szenen vor, die ähnliche Tags wie Ihre favorisierten Szenen haben

### Dashboard-Modul

Bietet ein interaktives Web-Dashboard mit:
- Visualisierungen der Statistiken
- Performer-Empfehlungen
- Szenen-Empfehlungen

### Updater-Modul

- Aktualisiert Performer mit EU-Cup-Größen-Tags
- Fügt Verhältnis-Informationen zu Performer-Details hinzu

### Discord-Modul

Sendet regelmäßige Updates an Discord:
- Statistik-Zusammenfassungen mit Diagrammen
- Performer-Empfehlungen
- Szenen-Empfehlungen

## GitHub-Nutzung

### Erste Schritte mit GitHub

1. **Repository erstellen**:
   - Besuchen Sie [GitHub](https://github.com) und melden Sie sich an
   - Klicken Sie auf das "+" in der oberen rechten Ecke und wählen Sie "New repository"
   - Geben Sie einen Namen ein (z.B. "stash-api-tools")
   - Wählen Sie "Public" oder "Private"
   - Klicken Sie auf "Create repository"

2. **Repository klonen**:
   ```bash
   git clone https://github.com/IHR_BENUTZERNAME/stash-api-tools.git
   cd stash-api-tools
   ```

3. **Dateien hinzufügen**:
   ```bash
   # Kopieren Sie alle Projektdateien in das geklonte Verzeichnis
   cp -r /Users/tench/Downloads/stash_api_project/* .
   
   # Fügen Sie die Dateien zum Git-Repository hinzu
   git add .
   
   # Erstellen Sie einen Commit
   git commit -m "Initiales Commit"
   
   # Pushen Sie die Änderungen zu GitHub
   git push origin main
   ```

### Regelmäßige Nutzung

1. **Änderungen überprüfen**:
   ```bash
   git status
   ```

2. **Änderungen hinzufügen**:
   ```bash
   git add .
   ```

3. **Änderungen committen**:
   ```bash
   git commit -m "Beschreibung der Änderungen"
   ```

4. **Änderungen hochladen**:
   ```bash
   git push origin main
   ```

5. **Änderungen herunterladen** (wenn Sie an verschiedenen Orten arbeiten):
   ```bash
   git pull origin main
   ```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

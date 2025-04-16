# Stash API Tools

Eine Sammlung von Tools zur Analyse und Verwaltung von Stash-Daten mit Fokus auf Performer-Statistiken, Empfehlungen und Integration.

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
pip install -e .
```

### Konfiguration

1. Klonen Sie das Repository oder laden Sie die Dateien herunter
2. Bearbeiten Sie die Datei `config/configuration.ini` und geben Sie Ihre Stash-API-Informationen und Discord-Webhook-URL ein

## Verwendung

### Statistiken generieren

```bash
python main.py stats
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

### Send Recommendations to Discord

```bash
python main.py discord
```

This command will:
- Generate performer recommendations
- Generate scene recommendations
- Send the recommendations to the configured Discord webhook
- Print recommendations to the console

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

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

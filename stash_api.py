import json
import requests
import re
import os
from datetime import datetime

# Prüfen, ob matplotlib installiert ist
try:
    import visualizations
    graphs_available = True
except ImportError:
    graphs_available = False
    print("\nHINWEIS: Das Visualisierungsmodul konnte nicht geladen werden.")
    print("Grafiken werden nicht erstellt.\n")

# Konfiguration
CONFIG = {
    "stash_url": "http://192.168.1.2:9999",
    "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiJrb3J0YWJhdG9yIiwic3ViIjoiQVBJS2V5IiwiaWF0IjoxNzQ0NTc4MDMxfQ.mqfwFbQKKhat5NCrNUXglh4uCu23iOPjCrgqOPwPoEs",
    "output_dir": "stats_graphs"
}

def safe_filename(filename):
    """Ersetzt ungültige Zeichen in Dateinamen"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def graphql_request(query, variables=None):
    """Führt eine GraphQL-Anfrage an die Stash-API aus"""
    headers = {
        "Content-Type": "application/json",
        "ApiKey": CONFIG["api_key"]
    }
    
    try:
        response = requests.post(
            f"{CONFIG['stash_url']}/graphql",
            headers=headers,
            data=json.dumps({
                "query": query,
                "variables": variables
            })
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result and result["errors"]:
                print(f"GraphQL Fehler: {result['errors']}")
                return None
            return result["data"]
        else:
            print(f"Fehler: Status Code {response.status_code}")
            return None
    except Exception as e:
        print(f"Fehler bei der Anfrage: {str(e)}")
        return None

def get_all_performers():
    """Ruft alle Performer von der Stash-API ab"""
    query = """
    query {
        allPerformers {
            id
            name
            birthdate
            gender
            country
            height_cm
            weight
            measurements
            scene_count
            rating100
            favorite
            tags {
                name
            }
            o_counter
        }
    }
    """
    
    data = graphql_request(query)
    if not data or not data["allPerformers"]:
        return []
    
    performers = data["allPerformers"]
    return [process_performer(p) for p in performers]

def process_performer(performer):
    """Verarbeitet und ergänzt die Daten eines Performers"""
    # Alter berechnen
    if performer.get("birthdate"):
        try:
            birth_date = datetime.strptime(performer["birthdate"], "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            performer["age"] = age
        except Exception:
            pass
    
    # Rating auf 5er-Skala
    if "rating100" in performer and performer["rating100"] is not None:
        performer["rating_5"] = round(performer["rating100"] / 20, 1)
    
    # Tag-Namen extrahieren
    if "tags" in performer:
        performer["tag_names"] = [tag["name"] for tag in performer.get("tags", [])]
    
    # BMI und BH-Größe berechnen
    calculate_bmi(performer)
    convert_bra_size(performer)
    calculate_metrics(performer)
    
    return performer

def calculate_bmi(performer):
    """Berechnet BMI, wenn Größe und Gewicht vorhanden sind"""
    height_cm = performer.get("height_cm")
    weight = performer.get("weight")
    
    if not (height_cm and weight and height_cm > 0 and weight > 0):
        return
    
    try:
        height_m = height_cm / 100
        weight_kg = float(weight)
        
        bmi = weight_kg / (height_m * height_m)
        performer["bmi"] = round(bmi, 1)
        
        # BMI-Kategorie
        if bmi < 18.5:
            performer["bmi_category"] = "Untergewicht"
        elif bmi < 25:
            performer["bmi_category"] = "Normalgewicht"
        elif bmi < 30:
            performer["bmi_category"] = "Übergewicht"
        else:
            performer["bmi_category"] = "Adipositas"
    except Exception:
        pass

def convert_bra_size(performer):
    """Konvertiert US/UK BH-Größen in deutsche Größen"""
    measurements = performer.get("measurements")
    if not measurements:
        return
    
    # Regex-Muster für BH-Größen
    match = re.search(r'(\d{2})([A-HJ-Z]+)', measurements)
    if not match:
        return
    
    us_band = int(match.group(1))
    us_cup = match.group(2)
    
    # Umrechnungstabellen
    band_conversion = {
        28: 60, 30: 65, 32: 70, 34: 75, 36: 80, 
        38: 85, 40: 90, 42: 95, 44: 100, 46: 105
    }
    
    cup_conversion = {
        "A": "A", "B": "B", "C": "C", "D": "D", 
        "DD": "E", "DDD": "F", "E": "E", "F": "F", 
        "G": "G", "H": "H"
    }
    
    cup_numeric = {
        "A": 1, "B": 2, "C": 3, "D": 4, 
        "E": 5, "DD": 5, "F": 6, "DDD": 6, 
        "G": 7, "H": 8
    }
    
    # Umrechnungen speichern
    de_band = band_conversion.get(us_band, round((us_band + 16) / 2) * 5)
    de_cup = cup_conversion.get(us_cup, us_cup)
    
    performer["german_bra_size"] = f"{de_band}{de_cup}"
    performer["us_bra_size"] = f"{us_band}{us_cup}"
    performer["cup_numeric"] = cup_numeric.get(us_cup, 0)

def calculate_metrics(performer):
    """Berechnet zusätzliche Kennzahlen"""
    cup_numeric = performer.get("cup_numeric", 0)
    bmi = performer.get("bmi", 0)
    height_cm = performer.get("height_cm")
    
    # Nur berechnen, wenn alle notwendigen Werte vorhanden sind
    if cup_numeric and cup_numeric > 0:
        if bmi and bmi > 0:
            performer["bmi_to_cup_ratio"] = round(bmi / cup_numeric, 2)
        
        if height_cm and height_cm > 0:
            performer["height_to_cup_ratio"] = round(height_cm / cup_numeric, 2)

def analyze_cup_size_distribution(performers):
    """Analysiert die Cup-Größen-Verteilung"""
    # Zähler initialisieren
    cup_counts = {}
    band_counts = {}
    combined_counts = {}
    numeric_cups = []
    
    # Cup-Umrechnungstabelle
    cup_numeric_to_letter = {
        1: "A", 2: "B", 3: "C", 4: "D", 
        5: "E", 6: "F", 7: "G", 8: "H"
    }
    
    # Daten sammeln
    for p in performers:
        if "german_bra_size" not in p or not p["german_bra_size"]:
            continue
        
        bra_size = p["german_bra_size"]
        try:
            match = re.match(r'(\d+)([A-HJ-Z]+)', bra_size)
            if not match:
                continue
                
            band = match.group(1)
            cup = match.group(2)
            
            cup_counts[cup] = cup_counts.get(cup, 0) + 1
            band_counts[band] = band_counts.get(band, 0) + 1
            combined_counts[bra_size] = combined_counts.get(bra_size, 0) + 1
            
            if p.get("cup_numeric"):
                numeric_cups.append(p["cup_numeric"])
        except Exception:
            continue
    
    total_analyzed = len(numeric_cups)
    
    if total_analyzed == 0:
        return "Keine Performer mit gültigen BH-Größen gefunden.", None
    
    # Sortierung für konsistente Ausgabe
    sorted_cups = sorted(cup_counts.keys(), key=lambda x: (len(x), x))
    sorted_bands = sorted(band_counts.keys(), key=lambda x: int(x))
    
    # Statistik berechnen
    mean_cup = sum(numeric_cups) / total_analyzed
    variance = sum((x - mean_cup) ** 2 for x in numeric_cups) / total_analyzed
    std_dev = variance ** 0.5
    
    most_common_cup = max(cup_counts.items(), key=lambda x: x[1])
    most_common_band = max(band_counts.items(), key=lambda x: x[1])
    most_common_combined = max(combined_counts.items(), key=lambda x: x[1])
    
    # Matrix erstellen
    matrix = []
    for band in sorted_bands:
        row = {"band": band}
        for cup in sorted_cups:
            size = f"{band}{cup}"
            row[cup] = combined_counts.get(size, 0)
        matrix.append(row)
    
    # Textausgabe erstellen
    result = []
    result.append("Cup-Größen Verteilung:")
    result.append("=====================\n")
    
    result.append("Cup-Größen Häufigkeit:")
    for cup in sorted_cups:
        count = cup_counts[cup]
        percentage = (count / total_analyzed) * 100
        result.append(f"{cup}: {count} ({percentage:.1f}%)")
    
    result.append("\nUnterbrustweiten Häufigkeit:")
    for band in sorted_bands:
        count = band_counts[band]
        percentage = (count / total_analyzed) * 100
        result.append(f"{band}: {count} ({percentage:.1f}%)")
    
    result.append("\nKombinierte BH-Größen Matrix:")
    header = "Band\\Cup | " + " | ".join(sorted_cups)
    result.append(header)
    result.append("-" * len(header))
    
    for row in matrix:
        row_str = f"   {row['band']}    | "
        for cup in sorted_cups:
            row_str += f" {row.get(cup, 0)}  | "
        result.append(row_str)
    
    result.append("\nStatistische Auswertung:")
    result.append(f"Durchschnittliche Cup-Größe (numerisch): {mean_cup:.2f}")
    avg_cup_letter = cup_numeric_to_letter.get(round(mean_cup), f"{cup_numeric_to_letter.get(int(mean_cup), '?')}-{cup_numeric_to_letter.get(int(mean_cup)+1, '?')}")
    result.append(f"Durchschnittliche Cup-Größe (Buchstabe): ca. {avg_cup_letter}")
    result.append(f"Standardabweichung: {std_dev:.2f}")
    result.append(f"Häufigste Cup-Größe: {most_common_cup[0]} ({most_common_cup[1]} Performer, {most_common_cup[1]/total_analyzed*100:.1f}%)")
    result.append(f"Häufigste Unterbrustweite: {most_common_band[0]} ({most_common_band[1]} Performer, {most_common_band[1]/total_analyzed*100:.1f}%)")
    result.append(f"Häufigste BH-Größe insgesamt: {most_common_combined[0]} ({most_common_combined[1]} Performer, {most_common_combined[1]/total_analyzed*100:.1f}%)")
    
    result.append("\nLegende der numerischen Cup-Werte:")
    legend = ", ".join([f"{num} = {letter}" for num, letter in cup_numeric_to_letter.items()])
    result.append(legend)
    
    # Daten für Visualisierung
    stats_data = {
        "cup_counts": cup_counts,
        "band_counts": band_counts,
        "combined_counts": combined_counts,
        "sorted_cups": sorted_cups,
        "sorted_bands": sorted_bands,
        "matrix": matrix,
        "mean_cup": mean_cup,
        "std_dev": std_dev,
        "numeric_cups": numeric_cups,
        "cup_numeric_to_letter": cup_numeric_to_letter
    }
    
    return "\n".join(result), stats_data

def filter_performers(performers, preferences):
    """Filtert Performer nach Benutzereinstellungen"""
    filtered = performers.copy()
    
    # Filtern nach Favoriten
    if preferences["only_favorites"]:
        filtered = [p for p in filtered if p.get("favorite")]
    
    # Filtern nach Mindest-Rating
    if preferences["min_rating"] > 0:
        filtered = [p for p in filtered if p.get("rating100") is not None and p.get("rating100") >= preferences["min_rating"]]
    
    # Filtern nach O-Counter
    if preferences["only_with_o_counter"]:
        filtered = [p for p in filtered if p.get("o_counter", 0) > 0]
    
    return filtered

def print_statistics(performers, title=""):
    """Gibt detaillierte Statistiken aus und erzeugt Visualisierungen"""
    if not performers:
        print("Keine Performer für die Statistik verfügbar.")
        return
    
    # Grundlegende Statistiken
    total_performers = len(performers)
    print(f"\n{'=' * 50}")
    print(f"STATISTIK: {title}")
    print(f"{'=' * 50}\n")
    print(f"Gesamtanzahl Performer in dieser Auswahl: {total_performers}")
    
    # Gruppierte Performer für verschiedene Statistiken
    complete_data = {
        "with_age": [p for p in performers if "age" in p],
        "with_scenes": [p for p in performers if p.get("scene_count", 0) > 0],
        "with_bmi": [p for p in performers if "bmi" in p],
        "with_cup": [p for p in performers if "german_bra_size" in p and p.get("cup_numeric", 0) > 0],
        "with_rating": [p for p in performers if p.get("rating100") is not None],
        "with_o_counter": [p for p in performers if p.get("o_counter", 0) > 0],
        "with_ratios": [p for p in performers if "bmi_to_cup_ratio" in p and "height_to_cup_ratio" in p]
    }
    
    # Verfügbarkeit der Daten
    print("\nVerfügbarkeit der Daten:")
    categories = {
        "with_age": "Alter", 
        "with_scenes": "Szenen",
        "with_bmi": "BMI",
        "with_cup": "Cup-Größe",
        "with_rating": "Bewertung",
        "with_o_counter": "O-Counter",
        "with_ratios": "BMI und Größe zu Cup-Verhältnis"
    }
    
    for key, label in categories.items():
        count = len(complete_data[key])
        percentage = count / total_performers * 100
        print(f"Performer mit {label}: {count} ({percentage:.1f}%)")
    
    # Durchschnittswerte
    print("\nBerechnete Durchschnittswerte:")
    
    # Alter
    if complete_data["with_age"]:
        avg_age = sum(p["age"] for p in complete_data["with_age"]) / len(complete_data["with_age"])
        print(f"- Durchschnittsalter: {avg_age:.1f} Jahre")
    
    # Szenen
    if complete_data["with_scenes"]:
        avg_scenes = sum(p["scene_count"] for p in complete_data["with_scenes"]) / len(complete_data["with_scenes"])
        max_scenes = max(p["scene_count"] for p in complete_data["with_scenes"])
        print(f"- Durchschnittliche Szenenanzahl: {avg_scenes:.1f}")
        print(f"- Maximale Szenenanzahl: {max_scenes}")
    
    # BMI
    if complete_data["with_bmi"]:
        avg_bmi = sum(p["bmi"] for p in complete_data["with_bmi"]) / len(complete_data["with_bmi"])
        print(f"- Durchschnittlicher BMI: {avg_bmi:.1f}")
        
        # BMI-Kategorien
        bmi_stats = {}
        for p in complete_data["with_bmi"]:
            category = p.get("bmi_category", "Unbekannt")
            bmi_stats[category] = bmi_stats.get(category, 0) + 1
        
        print("- BMI-Kategorien:")
        for category, count in bmi_stats.items():
            percentage = count / len(complete_data["with_bmi"]) * 100
            print(f"  - {category}: {count} ({percentage:.1f}%)")
    
    # Cup-Größe
    if complete_data["with_cup"]:
        avg_cup = sum(p["cup_numeric"] for p in complete_data["with_cup"]) / len(complete_data["with_cup"])
        print(f"- Durchschnittliche Cup-Größe (numerisch): {avg_cup:.1f}")
    
    # Rating
    if complete_data["with_rating"]:
        avg_rating = sum(p["rating100"] for p in complete_data["with_rating"]) / len(complete_data["with_rating"])
        print(f"- Durchschnittliche Bewertung: {avg_rating/20:.1f}/5 ({avg_rating:.1f}/100)")
        
        # Rating-Verteilung
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for p in complete_data["with_rating"]:
            stars = min(5, max(1, int(round(p["rating100"] / 20))))
            rating_dist[stars] = rating_dist.get(stars, 0) + 1
        
        print("- Rating-Verteilung:")
        for stars in sorted(rating_dist.keys()):
            count = rating_dist[stars]
            if count == 0:
                continue
            percentage = count / len(complete_data["with_rating"]) * 100
            print(f"  - {stars} Sterne: {count} ({percentage:.1f}%)")
    
    # Favoriten
    favorites_count = len([p for p in performers if p.get("favorite")])
    print(f"- Anzahl der Favoriten: {favorites_count} ({favorites_count/total_performers*100:.1f}%)")
    
    # O-Counter
    if complete_data["with_o_counter"]:
        o_values = [p["o_counter"] for p in complete_data["with_o_counter"]]
        avg_o_counter = sum(o_values) / len(o_values)
        max_o_counter = max(o_values)
        total_o_counter = sum(p.get("o_counter", 0) for p in performers)
        print(f"- Durchschnittlicher O-Counter: {avg_o_counter:.1f}")
        print(f"- Maximaler O-Counter: {max_o_counter}")
        print(f"- Gesamtsumme aller O-Counter: {total_o_counter}")
        
        # Top O-Counter-Werte
        o_counter_dist = {}
        for p in complete_data["with_o_counter"]:
            o_counter = p["o_counter"]
            o_counter_dist[o_counter] = o_counter_dist.get(o_counter, 0) + 1
        
        if o_counter_dist:
            print("- Top 5 O-Counter-Werte:")
            top_counters = sorted(o_counter_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            for counter, count in top_counters:
                percentage = count / len(complete_data["with_o_counter"]) * 100
                print(f"  - {counter} mal: {count} Performer ({percentage:.1f}%)")
    
    # Kennzahlen
    if complete_data["with_ratios"]:
        bmi_to_cup = [p["bmi_to_cup_ratio"] for p in complete_data["with_ratios"]]
        height_to_cup = [p["height_to_cup_ratio"] for p in complete_data["with_ratios"]]
        
        avg_bmi_cup = sum(bmi_to_cup) / len(bmi_to_cup)
        avg_height_cup = sum(height_to_cup) / len(height_to_cup)
        
        print(f"- Durchschnittlicher BMI zu Cup-Größe Verhältnis: {avg_bmi_cup:.2f}")
        print(f"- Durchschnittliche Größe zu Cup-Größe Verhältnis: {avg_height_cup:.2f}")
    
    # Cup-Größen Analyse
    if complete_data["with_cup"]:
        cup_analysis, stats_data = analyze_cup_size_distribution(performers)
        print(f"\n{cup_analysis}")

        # Ausgabe in Datei
        filename = f"cup_size_statistics_{safe_filename(title)}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"STATISTIK: {title}\n\n")
                f.write(cup_analysis)
            print(f"Detaillierte Cup-Größen Statistik wurde in '{filename}' gespeichert.")
            
            # Grafische Darstellungen
            if graphs_available and stats_data:
                visualizations.create_all_visualizations(performers, stats_data, title, CONFIG["output_dir"])
        except Exception as e:
            print(f"Fehler beim Speichern der Statistik: {e}")

def get_user_preferences():
    """Fragt Benutzereinstellungen ab"""
    print("\n==== STASH PERFORMER STATISTIK ====\n")
    
    # Performer-Auswahl
    while True:
        print("\nWelche Performer sollen analysiert werden?")
        print("1: Alle Performer")
        print("2: Nur Favoriten")
        choice = input("Bitte wähle (1/2): ").strip()
        if choice in ["1", "2"]:
            only_favorites = (choice == "2")
            break
        print("Ungültige Eingabe. Bitte 1 oder 2 wählen.")
    
    # Mindest-Rating
    while True:
        print("\nMindest-Rating für die Statistik:")
        print("1: Ab 3 Sterne")
        print("2: Ab 4 Sterne")
        print("3: Nur 5 Sterne")
        print("4: Alle Ratings einbeziehen")
        choice = input("Bitte wähle (1/2/3/4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            min_rating = {
                "1": 60,  # 3 Sterne = 60/100
                "2": 80,  # 4 Sterne = 80/100
                "3": 95,  # 5 Sterne = 95+/100
                "4": 0    # Alle
            }[choice]
            break
        print("Ungültige Eingabe. Bitte 1, 2, 3 oder 4 wählen.")
    
    # O-Counter
    while True:
        print("\nNur Performer mit O-Counter berücksichtigen?")
        print("1: Ja, nur Performer mit O-Counter > 0")
        print("2: Nein, alle Performer (unabhängig vom O-Counter)")
        choice = input("Bitte wähle (1/2): ").strip()
        if choice in ["1", "2"]:
            only_with_o_counter = (choice == "1")
            break
        print("Ungültige Eingabe. Bitte 1 oder 2 wählen.")
    
    return {
        "only_favorites": only_favorites,
        "min_rating": min_rating,
        "only_with_o_counter": only_with_o_counter
    }

def main():
    """Hauptfunktion"""
    # Benutzereinstellungen abfragen
    preferences = get_user_preferences()
    
    # Titel zusammenstellen
    title_parts = []
    if preferences["only_favorites"]:
        title_parts.append("Nur Favoriten")
    else:
        title_parts.append("Alle Performer")
    
    if preferences["min_rating"] == 60:
        title_parts.append("Ab 3 Sterne")
    elif preferences["min_rating"] == 80:
        title_parts.append("Ab 4 Sterne")
    elif preferences["min_rating"] == 95:
        title_parts.append("Nur 5 Sterne")
    
    if preferences["only_with_o_counter"]:
        title_parts.append("Mit O-Counter > 0")
    
    title = " - ".join(title_parts)
    
    print(f"\nHole Informationen für Performer ({title})...")
    
    # Performer abrufen und filtern
    all_performers = get_all_performers()
    
    if all_performers:
        filtered_performers = filter_performers(all_performers, preferences)
        
        if filtered_performers:
            print(f"\nErfolgreich {len(filtered_performers)} von {len(all_performers)} Performern nach den Kriterien gefiltert.")
            print_statistics(filtered_performers, title)
            
            if graphs_available:
                print(f"\nGrafische Statistiken wurden im Ordner '{CONFIG['output_dir']}' gespeichert.")
        else:
            print("Nach der Filterung sind keine Performer übrig geblieben. Bitte andere Filterkriterien wählen.")
    else:
        print("Es konnten keine Performer-Daten abgerufen werden.")

if __name__ == "__main__":
    main()
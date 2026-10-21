# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 1 / Sekwencje i sygnał mowy: od fali dźwiękowej do reprezentacji
**Autor / grupa:** Michał Pakuła / GL0.3
**Data:** 15.10.2026

---

## 1. Cel ćwiczenia

Celem ćwiczenia było prześledzenie drogi od nagrania mowy do reprezentacji cech, którą można podać modelowi sekwencyjnemu. Dla trzech własnych nagrań o rosnącej złożoności (samogłoska, wyraz, zdanie) wygenerowano przebieg czasowy, spektrogram log-mel oraz MFCC, porównano je ze sobą i zbadano wpływ parametrów `n_fft` (długość okna) oraz `hop_length` (krok ramkowania) na wygląd i wymiary reprezentacji.

## 2. Metoda i implementacja

### 2.1 Narzędzia i dane

- Python, `librosa`, `numpy`, `matplotlib`; środowisko Google Colab.
- Trzy własne nagrania w formacie m4a, wczytane z konwersją do 16 kHz mono (`librosa.load(..., sr=16000, mono=True)`):

| Nagranie | Plik | Czas [s] | Liczba próbek |
|---|---|---|---|
| samogłoska | `aaaaa.m4a` | 1,32 | 21 163 |
| wyraz | `wafel.m4a` | 1,05 | 16 726 |
| zdanie | `zjadłem_wafla.m4a` | 2,05 | 32 768 |

### 2.2 Konfiguracja

| Parametr | Wartość bazowa | Wartości w eksperymentach |
|---|---|---|
| `sr` | 16 000 Hz | – |
| `n_fft` (= `win_length`) | 512 (32 ms) | A: 256 / 512 / 1024 (16 / 32 / 64 ms) |
| `hop_length` | 160 (10 ms) | B: 80 / 160 / 320 (5 / 10 / 20 ms) |
| `n_mels` | 64 | – |
| `n_mfcc` | 13 | – |

W eksperymencie A stały jest `hop_length = 160`, a w eksperymencie B stałe jest `n_fft = 512`. Konfiguracja bazowa (512 / 160) jest punktem odniesienia (baseline) dla obu eksperymentów.

### 2.3 Najważniejsze fragmenty kodu

Log-mel i MFCC liczone są **ze wspólnej macierzy mel**, więc mają identyczną oś czasu i można je porównywać ramka po ramce. Wszystkie parametry przekazywane są jawnie.

```python
def licz_logmel(y, n_fft, hop, n_mels=N_MELS):
    """Log-mel w dB względem maksimum nagrania (0 dB = maksimum)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=n_fft, win_length=n_fft, hop_length=hop, n_mels=n_mels
    )
    return librosa.power_to_db(S, ref=np.max)


def licz_mfcc(S_db, n_mfcc=N_MFCC):
    """MFCC = DCT-II (ortho) z log-mel."""
    return librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)


def ramki_wzor(n_probek, hop):
    """Liczba ramek przy center=True: T = 1 + floor(N / hop)."""
    return 1 + n_probek // hop
```

Ścieżka przetwarzania: sygnał → `melspectrogram` (ramkowanie, okno Hanna, FFT, moc, filtry mel) → `power_to_db` (log-mel) → DCT (MFCC, 13 współczynników). Do prezentacji `specshow` otrzymuje faktyczny `hop_length`, a porównania tego samego nagrania wykonano na wspólnej skali barw (−80…0 dB).

## 3. Wyniki

### 3.1 Kontrola jakości nagrań

| Nagranie | Czas [s] | Peak | RMS | Przester |
|---|---|---|---|---|
| samogłoska | 1,32 | 0,498 | 0,1028 | nie |
| wyraz | 1,05 | 0,489 | 0,0801 | nie |
| zdanie | 2,05 | 0,474 | 0,0769 | nie |

### 3.2 Wymiary reprezentacji (n_fft = 512, hop = 160)

| Nagranie | Próbki | Log-mel | MFCC | T (wzór) |
|---|---|---|---|---|
| samogłoska | 21 163 | (64, 133) | (13, 133) | 133 |
| wyraz | 16 726 | (64, 105) | (13, 105) | 105 |
| zdanie | 32 768 | (64, 205) | (13, 205) | 205 |

Zmierzona liczba ramek zgadza się ze wzorem `T = 1 + ⌊N / hop⌋`. Dla modelu sekwencyjnego macierz transponuje się do układu `[T, cechy]`, np. `(205, 64)` dla zdania.

### 3.3 Porównanie nagrań

<img width="1508" height="711" alt="02_porownanie_nagran" src="https://github.com/user-attachments/assets/1cdba697-aac8-46aa-adaf-fa66ebb750fa" />


*Rys. 1. Log-mel (góra, wspólna skala dB) i MFCC (dół; oś pionowa to numer współczynnika, nie częstotliwość).*

Wykresy szczegółowe (przebieg + log-mel + MFCC) dla każdego nagrania:

<img width="989" height="889" alt="01_samogloska_przebieg_logmel_mfcc" src="https://github.com/user-attachments/assets/59d91753-b330-44da-b5dc-02f7518519da" />

<img width="989" height="889" alt="01_wyraz_przebieg_logmel_mfcc" src="https://github.com/user-attachments/assets/903df251-74f6-464a-bfaa-88cd16989fb6" />

<img width="989" height="889" alt="01_zdanie_przebieg_logmel_mfcc" src="https://github.com/user-attachments/assets/ce2d9e1b-ea16-4dae-b8c9-78f5be456128" />


*Rys. 2–4. Przebieg czasowy, log-mel i MFCC samogłoski, wyrazu i zdania.*

### 3.4 Eksperyment A: długość okna (`n_fft`)

| `n_fft` | Okno | Δf = sr / n_fft | Prążków na jedną harmoniczną (f0 ≈ 102 Hz) | Kontrast harmonicznych w widmie samogłoski* |
|---|---|---|---|---|
| 256 | 16 ms | 62,5 Hz | 1,6 | 2,5 dB |
| 512 | 32 ms | 31,25 Hz | 3,3 | 5,4 dB |
| 1024 | 64 ms | 15,6 Hz | 6,5 | 9,9 dB |

\* Odchylenie standardowe widma mocy w dB (0,3–1,0 s, pasmo 100–1000 Hz) od jego wygładzonej obwiedni (filtr medianowy ok. 300 Hz); im większe, tym wyraźniejsze prążki harmoniczne na tle obwiedni.

<img width="1508" height="1011" alt="03_ekspA_logmel_wszystkie" src="https://github.com/user-attachments/assets/0e100732-8993-4fdd-976f-a535cf984a35" />

*Rys. 5. Log-mel trzech nagrań przy `n_fft` = 256 / 512 / 1024.*

<img width="1508" height="461" alt="04_ekspA_stft_wybrane" src="https://github.com/user-attachments/assets/801e25c1-f573-4cdd-ab0a-fd388ce4ad5a" />

*Rys. 6. STFT zdania (0–4 kHz) przy trzech długościach okna.*

<img width="1508" height="461" alt="05_ekspA_mfcc_wybrane" src="https://github.com/user-attachments/assets/a486167c-3ed6-4971-88bc-56d7376b0900" />

*Rys. 7. MFCC zdania przy trzech wartościach `n_fft`.*

### 3.5 Eksperyment B: krok (`hop_length`)

Liczba ramek T (zmierzona = wzór `1 + ⌊N / hop⌋`):

| Nagranie | hop = 80 (5 ms) | hop = 160 (10 ms) | hop = 320 (20 ms) |
|---|---|---|---|
| samogłoska | 265 | 133 | 67 |
| wyraz | 210 | 105 | 53 |
| zdanie | 410 | 205 | 103 |

<img width="1508" height="1011" alt="06_ekspB_logmel_wszystkie" src="https://github.com/user-attachments/assets/f6fb0511-8f82-48ad-b3ba-58e877171433" />

*Rys. 8. Log-mel trzech nagrań przy `hop_length` = 80 / 160 / 320 (okno stałe, 32 ms).*

<img width="1508" height="461" alt="07_ekspB_zblizenie" src="https://github.com/user-attachments/assets/6f56e7b1-09b4-4dd0-94e4-6e832c2fd9df" />

*Rys. 9. Zbliżenie fragmentu 0,3–0,6 s zdania.*

## 4. Analiza i interpretacja

### 4.1 Oczekiwania a obserwacje

Oczekiwania pochodzą z materiałów pomocniczych do laboratorium.

| Oczekiwanie (materiały) | Obserwacja | Wynik |
|---|---|---|
| Dłuższe okno rozdziela harmoniczne samogłosek | Kontrast harmonicznych rośnie 2,5 → 5,4 → 9,9 dB dla 16 → 32 → 64 ms; na STFT zdania przy 256 harmoniczne są zlane, przy 1024 wyraźnie rozdzielone (Rys. 6) | potwierdzone |
| Dłuższe okno rozmywa szybkie zmiany w czasie | W badanych nagraniach nie udało się wykazać wyraźnej różnicy: pauza ok. 1,12–1,18 s w zdaniu (ok. 60 ms) jest widoczna przy każdym `n_fft` (Rys. 6). Przyczyną jest prawdopodobnie to, że zdarzenia w tych nagraniach są dłuższe lub porównywalne z oknem 64 ms | **niepotwierdzone** |
| Zmiana `hop_length` zmienia liczbę ramek, nie rozdzielczość | T: 401 / 201 / 101 dla 2 s; w nagraniach 265 / 133 / 67, 210 / 105 / 53, 410 / 205 / 103; na zbliżeniu (Rys. 9) mniejszy krok daje gęstszą siatkę tych samych ramek, bez ostrzejszych prążków | potwierdzone |
| Liczba ramek wynika ze wzoru `1 + ⌊N / hop⌋` | Zgodność we wszystkich 9 przypadkach | potwierdzone |

### 4.2 Samogłoska („aaaaa")

- Częstotliwość podstawowa jest stała: f0 ≈ 101–109 Hz (mediana 102 Hz), więc harmoniczne leżą co ok. 100 Hz.
- 71% energii mieści się poniżej 1 kHz; najsilniejsze pasma mel to ok. 575–1110 Hz.
- Na log-mel widać poziome prążki harmoniczne i obraz niemal niezmienny w czasie (Rys. 2); amplituda lekko opada ku końcowi nagrania.
- MFCC to poziome pasy: zmienność współczynników 1–12 w czasie jest minimalna (odchylenie standardowe ≤ 8). Struktury harmonicznej (f0) w MFCC nie widać. To zgodne z tym, że 13 pierwszych współczynników opisuje obwiednię widma, a nie prążki.

### 4.3 Wyraz („wafel")

Przebieg zawiera kilka wyraźnych zdarzeń (Rys. 3):

- ok. 0,11–0,19 s: energia o niskim widmie (centroid ok. 270 Hz);
- ok. 0,20–0,25 s: szum wysokoczęstotliwościowy (centroid ok. 3 kHz);
- ok. 0,26–0,40 s: samogłoska z energią do ok. 1,2 kHz;
- ok. 0,41–0,47 s: słaby szum (centroid ok. 2,5 kHz), prawdopodobnie głoska szczelinowa;
- ok. 0,50–0,68 s: druga samogłoska, energia głównie poniżej 1 kHz;
- po ok. 0,75 s: cisza.

94% energii leży poniżej 1 kHz. Na MFCC (Rys. 1) zmiany między tymi fragmentami widać jako przejścia barw w niskich współczynnikach (odchylenie standardowe c1 ≈ 34), ale pojedyncze pasy nie mają prostej interpretacji fizycznej.

### 4.4 Zdanie („zjadłem wafla")

- Wiele zmian w czasie: na przebiegu widać kilka oddzielnych odcinków o dużej amplitudzie rozdzielonych spadkami RMS (Rys. 4).
- ok. 0,16–0,22 s: silny szum (centroid ok. 4 kHz, ZCR 0,4–0,56);
- ok. 0,36–0,6 s: wyraźny łuk harmonicznych; f0 spada z ok. 143–148 Hz do ok. 105–107 Hz (Rys. 6);
- ok. 0,88–0,96 s: f0 wzrasta do ok. 166 Hz, wobec ok. 107–113 Hz w sąsiedztwie;
- ok. 1,12–1,18 s: przerwa między słowami (RMS ≈ 0,006–0,019), ciemny pas przez wszystkie pasma;
- ok. 1,36–1,40 s i ok. 1,56–1,60 s: krótkie odcinki szumowe (centroid ok. 1,9 kHz i ok. 4 kHz).

### 4.5 Czytelność MFCC a złożoność nagrania

Dla samogłoski MFCC jest czytelne (stałe pasy), dla wyrazu jeszcze da się wskazać przejścia, a dla zdania przechodzi w mozaikę, w której nie sposób przypisać pasów konkretnym głoskom (odchylenie standardowe c1 ≈ 41 wobec ≤ 8 dla samogłoski). Log-mel w tym samym zdaniu nadal pokazuje harmoniczne, formanty, szum i pauzy.

### 4.6 Ograniczenia i artefakty

- **Kodek stratny:** nagrania to m4a (AAC). Pauzy są niemal cyfrowym zerem (np. po 0,75 s w wyrazie), więc na wykresach są czarne (−80 dB); w nagraniu z szumem tła wyglądałyby inaczej.
- **Artefakt widmowy:** w samogłosce pasmo 6,5–7,2 kHz ma poziom ok. −74,5 dB wobec ok. −57,5 dB (5,5–6,5 kHz) i ok. −58 dB (7,2–8 kHz); ciemna pozioma linia przy ok. 7–8 kHz to najpewniej efekt kodeka lub resamplingu, a nie cecha mowy.
- **Skala względna:** dB liczone z `ref=np.max` są względne dla każdego nagrania, więc nie nadają się do porównywania bezwzględnej głośności nagrań.
- **Skala próby:** jeden mówca, trzy krótkie nagrania; wnioski dotyczą tych nagrań. Wartości f0 wyznaczono algorytmem pYIN i są wiarygodne głównie dla odcinków dźwięcznych.
- **Etykiety głosek:** opisy odcinków oparto na pomiarach (RMS, centroid widmowy, f0) i obrazie spektrogramu; nazwy głosek należy traktować jako interpretację.

## 5. Wnioski

1. Droga od nagrania do cech skraca sekwencję: sygnał 2,05 s to 32 768 próbek, a log-mel i MFCC mają po 205 ramek (64 i 13 wartości). Liczba ramek wynika dokładnie ze wzoru `1 + ⌊N / hop⌋`.
2. **Długość okna** decyduje o rozdzielczości częstotliwościowej: przy f0 ≈ 102 Hz okno 16 ms nie rozdziela harmonicznych (kontrast 2,5 dB), a okno 64 ms rozdziela je wyraźnie (9,9 dB). Efektu rozmycia czasowego nie udało się wykazać na tych nagraniach, bo ich zdarzenia nie są wystarczająco krótkie.
3. **Krok** zmienia tylko liczbę ramek (podwojenie kroku to połowa ramek) i nie poprawia rozdzielczości częstotliwościowej.
4. **Log-mel** zachowuje harmoniczne, formanty, szum i pauzy w całej gradacji nagrań. **MFCC** opisuje obwiednię widma i gubi strukturę harmoniczną; dla zdania przestaje być czytelne wzrokowo, co nie przeszkadza w użyciu go jako wejścia modelu.
5. **Kierunki poprawy:** nagrania w formacie bezstratnym (wav) i z udziałem kilku mówców, nagrania z szumem tła, dłuższe fragmenty zawierające krótkie zdarzenia (np. zwarcia p/t/k), które lepiej pokażą efekt rozmycia czasowego przy długim oknie.

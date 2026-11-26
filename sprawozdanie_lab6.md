# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 6 / Projekt generacyjny, część 1: przygotowanie danych dla Char-RNN
**Autor / grupa:** Michał Pakuła / GL0.3
**Data:** 26.11.2026

> **Informacja o użyciu AI.** Do redakcji analizy i wniosków, złożenia kodu (notebook) oraz przygotowania układu i formatowania sprawozdania użyto asystenta AI (Claude, Anthropic)

---

## 1. Cel ćwiczenia

Przygotowanie danych z polskiego korpusu do zadania przewidywania następnego znaku (Char-RNN, Laboratorium 7): wybór i oczyszczenie korpusu, podział na trening, walidację i test przed tworzeniem okien, słownik znaków zbudowany na treningu, `Dataset` i `DataLoader` dla każdego zbioru, kontrola batchy oraz zbadanie wpływu długości okna `SEKW` na liczbę par wejście–cel.

## 2. Metoda i implementacja

### 2.1 Korpus

- **Źródło:** „Lalka”, tom I, Bolesław Prus, plik TXT z serwisu Wolne Lektury (`https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt`).
- **Warunki użycia:** utwór jest w domenie publicznej. Stopka pliku odsyła do Zasad wykorzystania Wolnych Lektur (`https://wolnelektury.pl/info/zasady-wykorzystania/`) i wspomina m.in. Licencję Wolnej Sztuki 1.3 oraz zastrzeżenie praw do wydania krytycznego; tekst użyto wyłącznie do celów dydaktycznych w ramach laboratorium.
- **Uzasadnienie wyboru:** korpus jest wskazany w materiałach laboratorium, ma ok. 770 tys. znaków (duży w stosunku do okien rzędu 100 znaków), a polska proza dostarcza narracji, dialogów i bogatej odmiany wyrazów.
- **Charakterystyka:** XIX-wieczna powieść realistyczna; narracja i dialogi (zapisane z pauzą „—”), liczne wielokropki i cudzysłowy „ ”, archaiczne słownictwo i pisownia nazw; jeden autor i jeden tom.
- **Przygotowanie:** odczyt UTF-8, normalizacja Unicode do NFC, ujednolicenie końców linii (`\r\n` → `\n`), usunięcie ewentualnych znaczników HTML, **usunięcie stopki serwisu** (od frazy „Ta lektura, podobnie jak tysiące innych”), sprowadzenie wielokrotnych spacji do jednej i wielu pustych linii do jednej, `strip()`. Wielkość liter, interpunkcję i polskie znaki zachowano. Długość: 784 082 znaki przed oczyszczeniem, **772 639** po (−1,5%).

Fragment po oczyszczeniu (znaki 2000–2500, skrócony):

> „ właściciela, subiektów i chłopców przyczyny klęsk mających paść na S. Wokulskiego i jego sklep galanteryjny były tak jasne, jak gazowe płomyki oświetlające zakład. Przyczyny te tkwiły w niespokojnym charakterze, w awanturniczym życiu, zresztą w najświeższym postępku człowieka, który mając w ręku pewny kawałek chleba […]”

Najczęstsze znaki niealfanumeryczne po oczyszczeniu: spacja (117 800), `,` (10 656), `\n` (9 764), `.` (6 510), `—` (5 813), `…` (3 951), `?`, `!`, `:`, `„`, `;`, `”`.

### 2.2 Podział danych

Tekst podzielono **przed tworzeniem okien** na trzy ciągłe, rozłączne fragmenty w kolejności tekstu: pierwsze 80% (trening), kolejne 10% (walidacja), ostatnie 10% (test). Granice podziału: znaki 618 111 i 695 375 z 772 639. Proporcje 80/10/10 uzasadniają: korpus jest duży, więc walidacja i test po ok. 77 tys. znaków dają setki tysięcy niezależnie ocenianych znaków i wciąż mnóstwo par dla największego badanego okna (200), a trening zachowuje większość tekstu. Okna tworzone są wyłącznie wewnątrz każdego fragmentu (nie przekraczają granic podziału), żeby uniknąć przecieku: sąsiednie okna prawie się pokrywają (np. „Wiatr” i „iatr␣” mają 4 z 5 znaków wspólnych), więc losowe rozdzielenie gotowych okien dałoby niemal identyczne przykłady w treningu i walidacji. Podział zapisano (`lab6_dane.json`) na kolejne etapy projektu.

### 2.3 Słownik, Dataset i DataLoader

- Słownik znaków zbudowano **wyłącznie na zbiorze treningowym** (`stoi`, `itos` po posortowanym zbiorze różnych znaków) i użyto we wszystkich zbiorach. ID 0 zarezerwowano dla znaków spoza słownika (odkodowanie daje znak zastępczy `�`).
- `DatasetZnakowy`: wejście `x = dane[i : i + SEKW]`, cel `y = dane[i + 1 : i + SEKW + 1]` (przesunięcie o jeden znak), krok 1, `__len__ = max(0, N − SEKW)`.
- Osobny `Dataset` i `DataLoader` dla każdego zbioru: `batch_size = 64`; trening `shuffle=True`, `drop_last=True`; walidacja i test `shuffle=False`, `drop_last=False`. Ziarno: `torch.manual_seed(0)`.

```python
class DatasetZnakowy(Dataset):
    def __init__(self, dane, sekw):
        self.dane, self.sekw = dane, sekw
    def __len__(self):
        return max(0, len(self.dane) - self.sekw)
    def __getitem__(self, i):
        x = self.dane[i:i + self.sekw]
        y = self.dane[i + 1:i + self.sekw + 1]
        return torch.tensor(x), torch.tensor(y)
```

Kod: notebook `cwiczenie_lab6_sprawozdanie.ipynb` ([link do repozytorium](#)), oddany z wykonanymi komórkami.

## 3. Wyniki

### 3.1 Podział i liczebności (SEKW = 100, batch = 64)

| Zbiór | Dokumenty | Znaków | Udział [%] | Różnych znaków | Par | Batchy |
|---|---|---|---|---|---|---|
| trening | 1 | 618 111 | 80,0 | 101 | 618 011 | 9 656 |
| walidacja | 1 | 77 264 | 10,0 | 74 | 77 164 | 1 206 |
| test | 1 | 77 264 | 10,0 | 81 | 77 164 | 1 206 |

Liczba batchy w treningu to `⌊618 011 / 64⌋ = 9 656` (`drop_last=True`, odpada reszta 27 par), a w walidacji i teście `⌈77 164 / 64⌉ = 1 206` (`drop_last=False`).

### 3.2 Słownik i znaki spoza słownika

Rozmiar słownika: **V = 102** (101 różnych znaków z treningu + 1 ID na znak nieznany). W walidacji i teście **nie wystąpił żaden znak spoza słownika** (0 wystąpień), więc ID 0 jest zarezerwowane, ale niewykorzystane. Znaki słownika: biały znak końca linii i spacja, cyfry, interpunkcja (w tym `—`, `–`, `„`, `”`, `…`, `«`, `»`), litery łacińskie oraz polskie i kilka obcych (`à ç é ö ü Ó`).

### 3.3 Kontrola kodowania i batchy

Kodowanie i dekodowanie odtworzyło tekst źródłowy (fragmenty po 2 000 znaków w całości pokryte słownikiem) w treningu, walidacji i teście. Dla każdego zbioru pobrano batch: kształt wejść i celów **(64, 100)**, typ `torch.int64`, zakres ID mieści się w `0 … V−1 = 101` (trening 1–101, walidacja 1–101, test 2–96); w każdym batchu sprawdzono warunek `x[1:] == y[:-1]`. Dodatkowo sprawdzono parę o znanym indeksie `i = 1000` względem tekstu źródłowego (wejście = `tekst[i : i+100]`, cel = `tekst[i+1 : i+101]`, ostatni znak celu = `tekst[i+100]`):

| Zbiór | Wejście (początek) | Cel (początek) | Ostatni znak celu = znak źródła na pozycji 1100 |
|---|---|---|---|
| trening | `'urnikiem Wokulskiego; jedni krytykowali postępowanie prezyde…'` | `'rnikiem Wokulskiego; jedni krytykowali postępowanie prezyden…'` | `'s'` |
| walidacja | `'angbaum kupi dom Łęckich i czy, jak mówił Klejn, dolicytuje …'` | `'ngbaum kupi dom Łęckich i czy, jak mówił Klejn, dolicytuje g…'` | `'J'` |
| test | `'nasze pieniądze, to naturalnie musi spłacać nasze długi…”\n\n—'` | `'asze pieniądze, to naturalnie musi spłacać nasze długi…”\n\n— '` | `'t'` |

### 3.4 Wpływ długości okna SEKW

Ten sam tekst, podział i krok okna (1); `SEKW` ∈ {20, 50, 100, 200}:

| SEKW | Pary: trening | Pary: walidacja | Pary: test | Znaków-celów w treningu na epokę |
|---|---|---|---|---|
| 20 | 618 091 | 77 244 | 77 244 | 12 361 820 |
| 50 | 618 061 | 77 214 | 77 214 | 30 903 050 |
| 100 | 618 011 | 77 164 | 77 164 | 61 801 100 |
| 200 | 617 911 | 77 064 | 77 064 | 123 582 200 |

<img width="640" height="433" alt="image" src="https://github.com/user-attachments/assets/2e854617-7e10-4544-b9b2-ecac59b17edc" />


*Rys. 1. Liczba par wejście–cel w zależności od SEKW dla zbiorów treningowego, walidacyjnego i testowego (skala logarytmiczna; krzywe walidacji i testu pokrywają się).*

### 3.5 Częstość znaków w treningu

<img width="791" height="374" alt="image" src="https://github.com/user-attachments/assets/6f4c779a-8c56-4d46-9411-7f41d3449ccc" />


*Rys. 2. 20 najczęstszych znaków w zbiorze treningowym (␣ oznacza spację).*

### 3.6 Słowa (dla chętnych)

Najczęstsze słowa w treningu: *i* (2 849), *się* (2 560), *w* (1 984), *nie* (1 712), *na* (1 698), *z* (1 475), *do* (1 210), *że* (1 074), *a* (1 061), *to* (830), *pan* (691), *wokulski* (670), *za* (597), *o* (573), *ale* (538). Średnia długość słowa: 5,27 znaku, mediana 5.

<img width="635" height="394" alt="image" src="https://github.com/user-attachments/assets/f528a999-9eba-45b5-971d-436be3e19b35" />


*Rys. 3. Rozkład długości słów w zbiorze treningowym.*

## 4. Analiza i interpretacja

**Podział i słownik.** Walidacja i test mają mniej różnych znaków (74 i 81) niż trening (101), co jest zrozumiałe, bo rzadkie znaki (m.in. obce litery i cudzysłowy francuskie) występują tylko w treningu, a ponieważ słownik zbudowano na treningu, w walidacji i teście nie ma znaków spoza słownika. Ponieważ walidacja i test to ostatnie 20% tomu, zawierają późniejsze rozdziały (inne wydarzenia i bohaterowie niż w początku tomu), więc stanowią nieco trudniejszą, bardziej „przesuniętą” próbkę niż losowy wybór okien, ale bez przecieku.

**Częstość znaków.** Rozkład jest bardzo nierówny: spacja stanowi ok. 15,2% znaków treningu (93 888), a 20 najczęstszych znaków (spacja, `a`, `i`, `e`, `o`, `z`, `n`, `s`, `r`, `y`, `c`, `w`, `k`, `t`, `d`, `m`, `p`, `ł`, `u`, `l`) pokrywa ok. 83% tekstu, mimo że słownik ma 101 znaków. Wśród nich jest polskie `ł` (13 186). Model będzie więc widział znaki rzadkie (cyfry, `Ó`, `ç`, `«`) znacznie rzadziej, a prosta strategia wybierania częstych znaków da już niezerowy wynik, więc strata powinna być oceniana względem takiego punktu odniesienia.

**Długość okna a liczba par.** Przy kroku 1 liczba par wynosi `N − SEKW`, więc zmiana SEKW z 20 na 200 zmniejsza liczbę par w treningu o 180 (z 618 091 do 617 911, czyli o ok. 0,03%), a w walidacji i teście o 180 (ok. 0,23%), przez co krzywe na Rys. 1 są prawie płaskie, a krzywe walidacji i testu się pokrywają (oba zbiory mają tyle samo znaków). Natomiast liczba znaków-celów w treningu na epokę rośnie proporcjonalnie do SEKW (z 12,4 mln do 123,6 mln), więc dłuższe okno nie daje nowych danych, tylko dłuższy kontekst i większy koszt obliczeń i pamięci (tensory `[64, SEKW]`). Okna są silnie nakładające się: każdy znak występuje w ok. SEKW oknach.

**Długość słów a okno.** Średnia długość słowa 5,27 znaku (ze spacją ok. 6,3), więc okno SEKW = 100 obejmuje ok. 16 słów, okno 20 ok. 3 słów, a okno 200 ok. 32 słów. Rozkład długości słów jest nierówny, z najliczniejszymi słowami o 3 i 5 znakach i długim ogonem do kilkunastu znaków; krótkie słowa to głównie spójniki, przyimki i partykuły (*i*, *w*, *z*, *na*, *do*, *się*, *nie*), co potwierdza lista najczęstszych słów. Zgodnie z materiałami przy tej samej długości okna podział na znaki obejmuje mniej słów niż podział na słowa lub tokeny BPE, a dłuższy kontekst utrudnia rekurencyjnej sieci wykorzystanie odległej informacji.

**Ograniczenia.** Korpus to jeden tom jednego autora, więc model nauczy się stylu Prusa i słownictwa z tej powieści (w tym nazw własnych, np. „Wokulski”). W tekście pozostały drobne artefakty: nagłówek (tytuł i ISBN) na początku oraz linia `-----` na końcu, stanowiące znikomą część korpusu. Walidacja i test pochodzą z jednego, ciągłego końca tomu (jeden dokument), więc nie reprezentują całego tekstu. Statystyki słów policzono wyrażeniem `\w+`, które liczy też liczby jako słowa.

## 5. Wnioski

Z tekstu „Lalki” (tom I, 772 639 znaków po oczyszczeniu) przygotowano dane do Char-RNN: podział 80/10/10 na ciągłe, rozłączne fragmenty wykonano przed tworzeniem okien, a słownik V = 102 zbudowano na treningu; w walidacji i teście nie wystąpiły znaki spoza słownika. Kontrole (kodowanie i dekodowanie, kształt `(64, 100)`, typ `int64`, zakres ID, porównanie par z tekstem źródłowym, w tym ostatni znak celu) przeszły dla wszystkich trzech zbiorów. Rozkład znaków jest silnie nierówny (spacja ok. 15%, 20 znaków ok. 83% tekstu). Liczba par prawie nie zależy od SEKW (spadek o `SEKW`), ale koszt epoki (liczba znaków-celów) rośnie proporcjonalnie do SEKW, a okno 100 znaków obejmuje ok. 16 słów. Podział, słownik i ustawienia zapisano do użycia na Laboratorium 7; warto tam zmierzyć czas epoki dla różnych SEKW i porównać stratę z punktem odniesienia opartym na częstości znaków.

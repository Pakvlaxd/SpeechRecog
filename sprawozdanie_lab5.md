# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 5 / Wprowadzenie do NLP: reprezentacje tekstu i intuicja uwagi
**Autor / grupa:** [imię i nazwisko] / [grupa]
**Data:** 19.11.2026

> **Informacja o użyciu AI.** Do redakcji analizy i wniosków, przygotowania pomocniczej komórki kodu (pomiar tokenów na słowo dla różnych rozmiarów słownika BPE) oraz do przygotowania układu i formatowania sprawozdania użyto asystenta AI (Claude, Anthropic). Notebook laboratoryjny pochodzi od prowadzących. Wyniki i wykresy pochodzą z uruchomienia notebooka przez autora w Google Colab.

---

## 1. Cel ćwiczenia

Prześledzenie drogi od tekstu do reprezentacji liczbowych na małym korpusie polskim: tokenizacja podsłowna BPE (podziały form fleksyjnych, wpływ rozmiaru słownika na średnią liczbę tokenów na słowo), statyczne embeddingi word2vec (najbliżsi sąsiedzi, rzut PCA) oraz podstawowy mechanizm uwagi.

## 2. Metoda i implementacja

- Narzędzia: Python, `tokenizers` (BPE), `gensim` (Word2Vec), `scikit-learn` (PCA), NumPy, Matplotlib; Google Colab. Kod: notebook `cwiczenie_lab_05.ipynb` ([link do repozytorium](#)) z jedną dodaną komórką, oddany z wykonanymi komórkami.
- **Korpus:** 20 zdań po polsku o dwóch tematach (uczenie maszynowe i zwierzęta), powielone 5 razy (100 zdań) dla stabilniejszego treningu.
- **BPE:** `models.BPE` z `unk_token='[UNK]'`, pretokenizer `Whitespace`, `BpeTrainer(vocab_size=300, min_frequency=1)`; w eksperymencie z rozmiarem słownika limity 100, 200, 300, 400 i 500.
- **Word2Vec:** Skip-gram (`sg=1`), `vector_size=50`, `window=3`, `min_count=2`, 200 epok, `seed=0`, `workers=1`; słowa zapisane małymi literami, bez kropek.
- **PCA:** rzut wektorów wybranych słów na 2 składowe (`random_state=0`).
- **Uwaga (NumPy):** zapytanie to wektor słowa „znaczenie”, klucze i wartości to wektory słów zdania; skalowany iloczyn skalarny `K @ q / √d`, softmax, wektor kontekstu jako ważona suma.

Dodana komórka mierzy średnią liczbę tokenów na słowo na **tej samej próbce słów** (wszystkie wystąpienia słów w korpusie, bez interpunkcji; 715 wystąpień, 114 unikalnych słów) dla każdego rozmiaru słownika:

```python
PROBKA = [w for s in KORPUS for w in re.findall(r'\w+', s)]

def bpe(vs):
    t = Tokenizer(models.BPE(unk_token='[UNK]'))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(KORPUS, trainers.BpeTrainer(
        vocab_size=vs, special_tokens=['[UNK]'], min_frequency=1))
    return t

for vs in [100, 200, 300, 400, 500]:
    t = bpe(vs)
    sr = np.mean([len(t.encode(w).tokens) for w in PROBKA])
```

## 3. Wyniki

### 3.1 Podziały BPE form fleksyjnych (|V| = 300)

| Forma | Podział BPE |
|---|---|
| pisać | `pis` \| `a` \| `ć` |
| piszę | `p` \| `i` \| `sz` \| `ę` |
| napisany | `na` \| `pis` \| `any` |

### 3.2 Rozmiar słownika a średnia liczba tokenów na słowo

| Limit `vocab_size` | Faktyczny rozmiar słownika | Tokenów na słowo |
|---|---|---|
| 100 | 100 | 3,86 |
| 200 | 200 | 2,57 |
| 300 | 300 | 1,87 |
| 400 | 400 | 1,17 |
| 500 | 424 | 1,00 |

Dla limitu 500 tokenizer wyczerpał możliwe scalenia i uzyskał 424 wpisy, więc na wykresie użyto faktycznego rozmiaru.

![Tokeny na słowo a rozmiar słownika](wykresy/lab5_01_tokeny_na_slowo.png)

*Rys. 1. Średnia liczba tokenów na słowo w zależności od faktycznego rozmiaru słownika BPE.*

### 3.3 Najbliżsi sąsiedzi (Word2Vec, cosinus)

| Słowo | Najbliżsi sąsiedzi (cosinus) |
|---|---|
| [słowo 1] | [sąsiad 1 (0,xx), sąsiad 2 (0,xx), sąsiad 3 (0,xx)] |
| [słowo 2] | [...] |
| [słowo 3] | [...] |

### 3.4 Rzut PCA embeddingów

![Rzut PCA](wykresy/lab5_02_pca.png)

*Rys. 2. Embeddingi word2vec wybranych słów (rzut PCA 2D).*

### 3.5 Mechanizm uwagi (dodatkowo)

![Wagi uwagi](wykresy/lab5_03_uwaga.png)

*Rys. 3. Wagi uwagi dla zapytania „znaczenie” względem słów zdania „mechanizm uwagi waży znaczenie słów w zdaniu”.*

## 4. Analiza i interpretacja

### 4.1 Tokenizacja BPE i fleksja

Tokenizer uczy się na korpusie, więc granice tokenów wynikają z częstości par znaków, a nie z morfologii. W formach „pisać” i „napisany” wspólny fragment `pis` został wydzielony jako osobny token, co pozwala zapisać obie formy z użyciem tej samej części. Forma „piszę” ma inny temat (`pisz`, przemiana `s → sz`), więc jest dzielona na krótkie fragmenty (`p | i | sz | ę`) i nie dzieli z pozostałymi tokenu `pis`. Pokazuje to, że BPE wykorzystuje podobieństwo ortograficzne form, ale nie rozpoznaje alternacji tematu typowych dla polskiej fleksji. Granice tokenów nie pokrywają się też z granicami morfemów (np. dla „kotami” podział `ko | ta | mi` przecina temat „kot”).

### 4.2 Rozmiar słownika a długość tokenizacji

Średnia liczba tokenów na słowo spada monotonicznie wraz z rozmiarem słownika: od 3,86 przy 100 wpisach do 1,00 przy 424. Większy słownik mieści więcej wyuczonych scaleń, więc słowa są dzielone na mniej, dłuższych tokenów, a przy 424 wpisach każde słowo korpusu ma własny token. Płacimy za to większą tablicą osadzeń (|V| × d). **Ograniczenie:** pomiar wykonano na korpusie treningowym, więc wartość 1,00 jest optymistyczna; na nowym tekście, z innymi formami, średnia byłaby wyższa, a nieznane słowa byłyby składane z mniejszych części (lub zastępowane tokenem `[UNK]`, jeśli zawierają znak spoza zbioru widzianego w treningu).

### 4.3 Najbliżsi sąsiedzi

[Wpisz 2–4 zdania: czy sąsiedzi mają sens tematyczny, czy wynikają z częstych połączeń słów w zdaniach korpusu (sprawdź to w tekście korpusu), i przypomnij, że korpus jest mały (20 zdań powielonych 5 razy), więc wyniki są poglądowe.]

### 4.4 Rzut PCA

Na rzucie (Rys. 2) słowa związane z mową (`mowy`, `rozpoznawanie`, `synteza`) leżą blisko siebie w lewej części wykresu, a słowa związane z sieciami (`sieć`, `neuronowa`, `rekurencyjna`) tworzą grupę po prawej stronie. Słowa `kot` i `zwierzęta` są blisko siebie w środku wykresu, ale `pies` leży bliżej grupy `sieć`, `neuronowa` niż `kot`, więc temat zwierząt nie tworzy zwartego skupiska. Słowa `tekst`, `słowa` i `model` leżą w górnej części. Zgodnie z materiałami należy pamiętać, że PCA 2D pokazuje tylko dwa kierunki największej zmienności spośród 50 wymiarów, a ranking sąsiadów korzysta z cosinusa w pełnej przestrzeni, więc położenie na wykresie może dawać inne sąsiedztwo. Odchylenia (np. położenie słowa `pies`) są spodziewane przy tak małym korpusie.

### 4.5 Mechanizm uwagi

Najwyższą wagę dla zapytania „znaczenie” ma samo słowo „znaczenie” (ok. 0,18), co jest naturalne, bo iloczyn skalarny wektora z samym sobą jest duży, kolejne są „waży” i „słów” (ok. 0,16), a najniższą waga słowa „w” (ok. 0,085). Wagi są dość wyrównane (0,085–0,18), co jest spodziewane przy statycznych embeddingach z małego korpusu, gdzie wektory są do siebie dość podobne. W uproszczeniu z notebooka macierz wartości jest tą samą macierzą co klucze, więc wektor kontekstu to ważona suma osadzeń słów zdania.

### 4.6 Ograniczenia

- Korpus ma tylko 20 różnych zdań, więc embeddingi i sąsiedzi są poglądowe.
- Pomiar tokenów na słowo wykonano na próbce będącej korpusem treningowym.
- BPE zna wyłącznie znaki widziane w korpusie; formy z innymi znakami dałyby `[UNK]`.
- PCA 2D nie oddaje pełnej przestrzeni 50 wymiarów.

## 5. Wnioski

BPE wydziela wspólne fragmenty form tego samego wyrazu (`pis` w „pisać” i „napisany”), ale granice tokenów nie odpowiadają morfemom i nie obejmują przemian tematu, jak w „piszę”. Rozmiar słownika BPE silnie wpływa na długość tokenizacji: średnia liczba tokenów na słowo spada z 3,86 (100 wpisów) do 1,00 (424 wpisy), kosztem większej tablicy osadzeń, a wartość 1,00 dotyczy próbki treningowej. Embeddingi word2vec na małym korpusie grupują słowa o wspólnym kontekście (np. słowa związane z mową leżą obok siebie), ale temat zwierząt nie jest wyraźnie wydzielony, a [wpisz wniosek z sąsiadów]. Mechanizm uwagi daje wagi sumujące się do 1 i ważoną sumę wektorów, jednak przy statycznych osadzeniach z małego korpusu wagi są dość wyrównane. Wnioski dotyczą jednego małego korpusu; kierunki dalszej pracy to większy korpus, ocena na tekście spoza korpusu i porównanie z tokenizerem HerBERT.

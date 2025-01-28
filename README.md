

Opis
Ten projekt wykorzystuje dane kredytowe z niemieckiego banku do przewidywania ryzyka kredytowego (dobry/zły kredyt) za pomocą modelu Random Forest Classifier. Głównym celem jest zbudowanie modelu klasyfikacyjnego i ocena jego skuteczności.

Funkcjonalności
Wstępne przetwarzanie danych (mapowanie wartości kategorycznych, usuwanie braków danych).
Budowanie modelu Random Forest z hiperparametrami.
Ocena modelu za pomocą:
Dokładności (accuracy),
Raportu klasyfikacji (classification report).
Predykcja ryzyka kredytowego dla nowych danych.
Dane wejściowe
Dane klientów, takie jak wiek, płeć, cel kredytu, kwota kredytu, oszczędności, itp.
Wymagania
Python
Biblioteki: pandas, numpy, scikit-learn
Sposób działania
Wczytanie i wstępne przetwarzanie danych:

Mapowanie wartości tekstowych na wartości numeryczne.
Przygotowanie danych do modelu.
Trenowanie modelu:

Random Forest z wybranymi hiperparametrami.
Ocena modelu:

Obliczenie dokładności i analiza ważności cech.
Predykcja:

Przewidywanie ryzyka kredytowego dla nowych klientów.
Wyniki
Model ocenia, czy dany kredyt jest "dobry" (niski poziom ryzyka) czy "zły" (wysokie ryzyko), opierając się na cechach klientów. Wynik jest prezentowany jako przewidywana etykieta.

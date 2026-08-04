# FALCON — Guida all'esecuzione degli esperimenti

Questa guida contiene solo le istruzioni per eseguire gli esperimenti. Non serve
altro contesto: tutti gli esperimenti di questo pacchetto usano dati sintetici,
quindi NON è necessario scaricare alcun dataset.

## 1. Installazione dell'ambiente (una sola volta)

Serve una installazione di conda (Miniconda va benissimo).

```bash
conda env create -f environment.yml
conda activate falcon
pytest        # se tutti i test passano, l'ambiente è pronto
```

## 2. Esperimento A — matrice delle regole di aggregazione (pochi minuti)

```bash
python experiments/run_e5_aggregators.py
```

Esegue 3 regole di aggregazione x 3 tipi di guasto e produce
`results/e5_aggregators/summary.json` e `e5_table.md`.

## 3. Esperimento B — guasti composti (pochi minuti)

```bash
python experiments/run_compound.py
```

Inietta guasti in DUE stadi contemporaneamente e verifica come il framework
riporta l'ambiguità. Produce `results/compound/summary.json` e un report per caso.

## 4. Invio dei risultati

Al termine di entrambi gli esperimenti eseguire:

```bash
python scripts/collect_output.py
# -> crea tmp/Output_YYYY-MM-DD_HH-MM-SS.zip
```

Inviare il file zip generato. Aggiungere `--full` SOLO se viene richiesto
esplicitamente (include i tensori grezzi, molto pesanti).

## In caso di problemi

- `pytest` fallisce: probabile problema di versioni Python/dipendenze —
  eseguire `conda env remove -n falcon` e ripartire dal punto 1.
- Un esperimento si interrompe: rieseguire il comando; poi inviare comunque lo
  zip di `collect_output.py` insieme al messaggio di errore completo.

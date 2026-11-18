# Bucle de mutación — instrucciones para el agente

Lee también `CLAUDE.md` y `AGENTS.md`: sus reglas siguen valiendo aquí.

## Objetivo

Subir el porcentaje de mutantes muertos del módulo `target_module` de `loop/config.json`
**añadiendo tests**. Un mutante es una copia del módulo con un cambio pequeño (un `<` por `<=`,
un `255` por `256`, un `"a"` por `"A"`…). Está «muerto» si algún test falla con él; si sobrevive,
ningún test notaría ese fallo en el código real.

Métrica (D039): `(killed + timeout + segfault) / (total − skipped)`. «no tests» cuenta como vivo.

## Qué puedes tocar y qué no

- **Sí:** ficheros dentro de `tests/` y `loop/RESUMEN.md`.
- **No:** `src/`, `configs/`, `data/`, `pyproject.toml`, `requirements*.txt` ni ningún otro fichero
  de `loop/`. El código de `src/` es de **solo lectura**: si un test descubre un fallo real en
  `src/`, no lo arregles; anótalo en `loop/RESUMEN.md`.
- `loop/evaluate.py` comprueba los ficheros protegidos contra el tag base antes de cada
  experimento. Si encuentra un cambio, escribe `loop/STOP` y el bucle termina.
- No hagas commit, stash, reset, checkout ni push: de eso se encarga `loop/evaluate.py`.

## Cómo trabajar

1. **Línea base.** Lo primero, sin tocar nada: `python loop/evaluate.py "línea base"`.
   Después lee `loop/survivors.txt`: cada entrada da `src/<módulo>.py:línea`, el estado, el nombre
   del mutante y la línea original (`-`) frente a la mutada (`+`).
2. **Un experimento = un test o un grupo pequeño de tests** que ataque supervivientes concretos
   (mejor varios de la misma función). Escríbelos en `tests/`, en un fichero existente del módulo
   o en `tests/test_<módulo>_mutation.py`.
3. **Evalúa:** `python loop/evaluate.py "<qué test añade y qué fallo caza>"`. El juez pasa ruff y
   pytest, mide, y se queda con el cambio solo si mejora más de `min_improvement`; si no, lo
   guarda en un stash y deja `tests/` como estaba. Lee la línea que imprime (estado y descartes
   seguidos) y vuelve a leer `loop/survivors.txt`.
4. Repite hasta que el hook te deje parar (límite de iteraciones, descartes seguidos o `STOP`).

## Cómo deben ser los tests

- **Solo datos sintéticos:** `tmp_path`, imágenes generadas con NumPy/OpenCV. Nunca los datasets
  de `data/`.
- **Cada test con un docstring** que diga qué fallo de visión por computador caza. Ejemplos:
  - canales BGR/RGB intercambiados (OpenCV lee en BGR);
  - H de OpenCV en 0–179, no en 0–360;
  - desbordes de `uint8` (`200 + 100` da `44`, no `300`);
  - máscaras con valores distintos de 0/255, o vacías;
  - umbrales con `<` frente a `<=` en el borde exacto;
  - canales de Lab desplazados (a*, b* centrados en 128).
- **Nada de asserts triviales** (`assert result is not None`, `assert len(x) > 0` sin más):
  compara contra el valor que se espera en un caso construido a mano, con una explicación de por
  qué ese valor es el correcto.
- Tests pequeños, deterministas (semilla fija si hay aleatoriedad) y rápidos (menos de 1 s).

## Mutantes equivalentes

Algunos mutantes no cambian el comportamiento (p. ej. cambiar el texto de un mensaje de error
que nadie compara, o `x * 1.0` en un contexto donde da igual). **No fuerces tests** para ellos:
apúntalos en `loop/RESUMEN.md` con su `src/<módulo>.py:línea` y una frase de por qué son
equivalentes o no merecen un test.

## Al terminar

Escribe `loop/RESUMEN.md` con:

- valor base frente a valor final (de `loop/results.tsv`) y número de experimentos keep/discard;
- qué tests has añadido y qué fallo caza cada uno;
- los supervivientes que quedan, agrupados por función, y qué haría falta para matarlos;
- los mutantes equivalentes y cualquier fallo real de `src/` que hayas visto (sin arreglarlo).

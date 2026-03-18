# Manual de Usuario — RevFast

**Sistema de Apoyo a la Decisión para la Pre-Evaluación de Presentaciones de Video**
Centro Institucional de Lenguas (CIL) — Universidad Autónoma de Yucatán (UADY)

> **Nota importante:** El análisis generado por RevFast es una sugerencia preliminar. El docente conserva en todo momento la autoridad final sobre la calificación del alumno.

---

## 1. ¿Qué es RevFast?

RevFast es un Sistema de Apoyo a la Decisión (DSS, por sus siglas en inglés) basado en inteligencia artificial multimodal. Fue desarrollado para el CIL de la UADY con el objetivo de ayudar a los docentes a **pre-evaluar** presentaciones orales de estudiantes grabadas en video.

El sistema analiza automáticamente:
- **Audio** — transcribe el habla del estudiante y evalúa vocabulario, gramática, pronunciación y fluidez.
- **Video** — detecta el lenguaje corporal (contacto visual, postura, gestos) usando visión por computadora.

El resultado es un reporte estructurado con puntajes por criterio, fortalezas, áreas de mejora y un resumen en lenguaje natural. Este reporte sirve como punto de partida para la revisión del docente, **no como calificación definitiva**.

---

## 2. Requisitos del Sistema

| Componente | Mínimo recomendado |
|---|---|
| Python | 3.11 o superior |
| Sistema operativo | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| RAM | 4 GB (8 GB recomendado para Whisper `medium` o `large`) |
| CPU | Moderno de 64 bits; GPU no requerida pero acelera Whisper |
| Espacio en disco | ~3 GB (modelos de Whisper + MediaPipe) |
| Conexión a internet | Requerida para la API de Gemini y la primera descarga de modelos |

---

## 3. Instalación

### 3.1 Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd cilcode
```

### 3.2 Crear un entorno virtual (recomendado)

```bash
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En macOS/Linux:
source .venv/bin/activate
```

### 3.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3.4 Primera descarga de modelos

La primera vez que se ejecute el pipeline, RevFast descargará automáticamente los modelos de MediaPipe (~2 MB) a `~/.cache/video_grader_ai/models/`. Esto requiere conexión a internet y puede tardar unos segundos.

Los modelos de Whisper se descargan también en la primera ejecución según el tamaño elegido (`base` ≈ 145 MB, `small` ≈ 480 MB, `medium` ≈ 1.5 GB, `large` ≈ 3 GB).

---

## 4. Configuración de la API (Gemini)

RevFast utiliza el modelo `gemini-2.5-pro` de Google para generar la evaluación. Para esto se necesita una clave de API.

### 4.1 Obtener la clave

1. Ir a [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Iniciar sesión con una cuenta de Google.
3. Crear un nuevo proyecto o seleccionar uno existente.
4. Generar una clave de API y copiarla.

### 4.2 Configurar la clave

En la raíz del proyecto, crear un archivo `.env` a partir del ejemplo:

```bash
cp .env.example .env
```

Editar `.env` y reemplazar el valor de la clave:

```
GOOGLE_API_KEY=TU_CLAVE_AQUI
```

> **Seguridad:** Nunca compartas ni subas al repositorio el archivo `.env` con tu clave real.

---

## 5. Uso Básico

### 5.1 Sintaxis del comando (CLI)

```bash
python main.py VIDEO_PATH [--whisper-model {tiny,base,small,medium,large}]
                          [--sample-every N]
                          [--rubric RUBRIC_PATH]
                          [--save]
```

### 5.2 Sintaxis de la aplicación con interfaz en Web
```
python web/app.py
# por defecto, la aplicación se abre en el puerto 5005
# en caso de recibir error, modificar en el main de app.py
# Abre http://127.0.0.1:5005
```
Se propone una interfaz web con el fin de facilitar la visualización y respuesta del modelo,
pensado principalmente para personas con vago o nulo conocimiento en CLI.

### 5.3 Opciones disponibles

| Opción | Tipo | Default | Descripción |
|---|---|---|---|
| `VIDEO_PATH` | string | (requerido) | Ruta al archivo de video (mp4, avi, mov, etc.) |
| `--whisper-model` | choice | `base` | Tamaño del modelo de Whisper. Modelos más grandes son más precisos pero más lentos. |
| `--sample-every N` | entero | `15` | Procesar MediaPipe cada N fotogramas. Valores más bajos = mayor precisión, mayor tiempo. |
| `--rubric RUBRIC_PATH` | string | Elementary 1 CIL | Ruta a un archivo JSON de rúbrica personalizada. |
| `--save` | flag | desactivado | Guardar el resultado como `<video>_evaluation.json` junto al video. |

### 5.4 Ejemplos

**Evaluación básica con opciones predeterminadas:**
```bash
python main.py presentacion_alumno.mp4
```

**Con modelo Whisper más preciso y guardado del resultado:**
```bash
python main.py presentacion_alumno.mp4 --whisper-model small --save
```

**Con rúbrica personalizada para otro nivel:**
```bash
python main.py presentacion_alumno.mp4 --rubric rubrics/mi_nivel.json --save
```

**Muestreo de video más denso (cada 5 fotogramas):**
```bash
python main.py presentacion_alumno.mp4 --sample-every 5
```

### 5.5 Ejemplo de salida esperada

```
[1/3] Transcribing audio with Whisper (base) …
      Language detected: en
      Transcript (312 chars): Hello, today I'm going to talk about my favorite hobby...

[2/3] Analysing video with MediaPipe (every 15 frames) …
      Frames sampled: 180
      Face detected: 172 frames
      Pose detected: 168 frames
      Avg attention score: 0.83

[3/3] Requesting evaluation from Gemini 1.5-pro …

=== Evaluation Result ===
{
  "vocabulario": 4,
  "precision_gramatical": 3,
  "pronunciacion_fluidez": 4,
  "contenido_organizacion": 5,
  "lenguaje_corporal_tiempo_preparacion": 4,
  "puntuacion_total": 20,
  "aprobado": true,
  "fortalezas": [
    "Buen contacto visual con la cámara",
    "Vocabulario apropiado para el nivel"
  ],
  "areas_mejora": [
    "Mejorar la precisión gramatical en tiempos pasados"
  ],
  "resumen": "El alumno demostró un buen dominio del tema con vocabulario adecuado. La organización fue clara y coherente. Se recomienda trabajar en la precisión gramatical."
}
```

---

## 6. Rúbricas Personalizadas

### 6.1 ¿Cuándo usar una rúbrica personalizada?

RevFast incluye la rúbrica oficial del CIL Elementary 1 como predeterminada. Si deseas evaluar presentaciones de otro nivel (Elementary 2, Intermediate, etc.) o con criterios distintos, puedes crear y usar tu propio archivo de rúbrica en formato JSON.

### 6.2 Estructura del archivo JSON

```json
{
  "nivel": "Elementary 2",
  "idioma": "en",
  "escala_min": 1,
  "escala_max": 5,
  "umbral_aprobatorio": 15,
  "descripcion_escala": {
    "5": "Completo dominio / excelente",
    "4": "Dominio adecuado / muy bueno",
    "3": "Dominio inadecuado / algo adecuado",
    "2": "Muy poco dominio / débil",
    "1": "Ningún dominio / muy débil"
  },
  "criterios": [
    {
      "clave": "vocabulario",
      "nombre": "Vocabulario",
      "descripcion": "Control del vocabulario requerido para el nivel Elementary 2."
    },
    {
      "clave": "fluidez",
      "nombre": "Fluidez",
      "descripcion": "Capacidad de hablar con ritmo natural y pocas pausas."
    }
  ]
}
```

### 6.3 Tabla de campos requeridos

| Campo | Tipo | Descripción |
|---|---|---|
| `nivel` | string | Nombre del nivel (ej. `"Elementary 2"`) |
| `idioma` | string | Código de idioma de la presentación (`"en"` o `"es"`) |
| `escala_min` | entero | Puntuación mínima por criterio (ej. `1`) |
| `escala_max` | entero | Puntuación máxima por criterio (ej. `5`); debe ser > `escala_min` |
| `umbral_aprobatorio` | entero | Puntuación total mínima para aprobar (exclusivo: el alumno aprueba si `puntuacion_total > umbral`) |
| `descripcion_escala` | objeto | Una entrada por cada nivel entero en `[escala_min, escala_max]`; las claves son strings (`"1"`, `"2"`, …) |
| `criterios` | array | Lista no vacía de objetos `{clave, nombre, descripcion}` |

**Reglas para `clave`:**
- Solo letras minúsculas, dígitos y guión bajo: `^[a-z][a-z0-9_]*$`
- Debe comenzar con una letra
- Debe ser única dentro de la rúbrica
- Ejemplos válidos: `vocabulario`, `fluidez2`, `lenguaje_corporal`
- Ejemplos inválidos: `Vocabulario`, `fluidez oral`, `2fluency`

### 6.4 Errores de validación comunes

| Error | Causa | Solución |
|---|---|---|
| `faltan campos requeridos` | Algún campo obligatorio no está presente | Revisar que el JSON incluya todos los campos de la tabla anterior |
| `escala_min debe ser menor que escala_max` | Se configuró `escala_min >= escala_max` | Corregir los valores de escala |
| `descripcion_escala no cubre el nivel N` | Falta la descripción para algún nivel numérico | Agregar la entrada faltante en `descripcion_escala` |
| `criterios no puede estar vacío` | El array `criterios` está vacío | Agregar al menos un criterio |
| `clave duplicada` | Dos criterios tienen la misma clave | Usar claves únicas para cada criterio |
| `no cumple el formato` | La clave contiene mayúsculas, espacios u otros caracteres no permitidos | Reescribir la clave en minúsculas con guiones bajos |
| `umbral_aprobatorio fuera del rango posible` | El umbral no cabe en el rango `[escala_min*n, escala_max*n]` | Ajustar el umbral según el número de criterios y la escala |

---

## 7. Interpretación del Resultado

### 7.1 Significado de cada campo

| Campo | Tipo | Descripción |
|---|---|---|
| `vocabulario` | 1–5 | Puntuación del criterio de vocabulario |
| `precision_gramatical` | 1–5 | Puntuación del criterio gramatical |
| `pronunciacion_fluidez` | 1–5 | Puntuación de pronunciación y fluidez |
| `contenido_organizacion` | 1–5 | Puntuación del contenido y organización |
| `lenguaje_corporal_tiempo_preparacion` | 1–5 | Puntuación del lenguaje corporal y preparación |
| `puntuacion_total` | 5–25 | Suma de los cinco criterios anteriores |
| `aprobado` | booleano | `true` si `puntuacion_total > 15` (umbral predeterminado) |
| `fortalezas` | lista de strings | Aspectos positivos observados en la presentación |
| `areas_mejora` | lista de strings | Aspectos que el alumno debe mejorar |
| `resumen` | string | Resumen en 2–4 oraciones de la evaluación global |

### 7.2 Umbral de aprobación

Con la rúbrica predeterminada (5 criterios, escala 1–5):
- **Rango posible:** 5–25 puntos
- **Umbral de aprobación:** estrictamente mayor que 15
- **Puntuaciones ≤ 2** por criterio son señal de dominio muy débil

### 7.3 Naturaleza cualitativa de la retroalimentación

Los campos `fortalezas`, `areas_mejora` y `resumen` son generados por un modelo de lenguaje y tienen carácter **orientativo**. Pueden contener imprecisiones, especialmente cuando la calidad de audio o video es baja. El docente debe revisarlos críticamente antes de compartirlos con el alumno.

### 7.4 Carácter preliminar del análisis

RevFast es una herramienta de pre-evaluación, no un sistema de calificación automática. Su función es reducir el tiempo de revisión inicial del docente, proporcionando un primer análisis que el docente debe validar, ajustar y aprobar antes de comunicar resultados al alumno.

---

## 8. Limitaciones Conocidas

- **Duración del video:** optimizado para presentaciones de 1 a 5 minutos. Videos más largos pueden incrementar significativamente el tiempo de procesamiento.
- **Pre-evaluación, no calificación:** la IA sugiere puntajes; la calificación final corresponde siempre al docente.
- **Requiere conexión a internet:** la evaluación de Gemini no funciona sin acceso a la API de Google.
- **Intensivo en CPU:** el modelo Whisper `medium` o `large` puede tardar varios minutos en equipos sin GPU.
- **Solo interfaz de línea de comandos:** actualmente no hay interfaz gráfica (GUI) ni aplicación web.
- **Dependencia de la calidad del audio:** ruido de fondo, música o solapamiento de voces reducen la precisión de la transcripción.
- **Dependencia de la calidad del video:** iluminación deficiente u oclusión del rostro reducen la precisión de la detección de lenguaje corporal.

---

## 9. Preguntas Frecuentes

**¿RevFast reemplaza al docente?**
No. RevFast es una herramienta de apoyo. El docente tiene la última palabra en todas las decisiones de evaluación.

**¿Mis datos de alumnos se almacenan en la nube?**
El video y la transcripción se envían a la API de Gemini de Google para generar la evaluación. Consulta la política de privacidad de Google AI para más detalles. RevFast en sí mismo no almacena datos en ningún servidor externo.

**¿Funciona con videos en español?**
Sí. Whisper detecta el idioma automáticamente. Para la rúbrica predeterminada, el idioma esperado de la presentación es inglés (`"idioma": "en"`), pero puedes crear una rúbrica personalizada con `"idioma": "es"` para presentaciones en español.

**¿Qué pasa si el alumno habla muy bajo o hay mucho ruido?**
Whisper intentará transcribir el audio de todas formas, pero la calidad de la transcripción puede ser baja, lo que afectará la evaluación. Se recomienda usar videos con audio claro y sin ruido de fondo.

**¿Puedo usar RevFast con otros niveles del CIL además de Elementary 1?**
Sí. Crea un archivo JSON de rúbrica para el nivel deseado (ver sección 6) y pásalo con la opción `--rubric`.

---

## 10. Solución de Problemas

| Error / Mensaje | Causa probable | Solución |
|---|---|---|
| `GOOGLE_API_KEY is not set` | No se configuró la clave de API | Crear el archivo `.env` con la clave (ver sección 4) |
| `FileNotFoundError: [video]` | Ruta al video incorrecta o archivo inexistente | Verificar la ruta y que el archivo exista |
| `FileNotFoundError: [rubric]` | Ruta a la rúbrica incorrecta | Verificar la ruta con `--rubric` |
| `El archivo de rúbrica no es JSON válido` | El archivo JSON tiene errores de sintaxis | Abrir el archivo en un editor con validación de JSON |
| `Rúbrica inválida: faltan campos requeridos` | Campos obligatorios ausentes en el JSON de rúbrica | Revisar la sección 6.3 |
| `Gemini returned non-JSON response` | La API de Gemini devolvió una respuesta inesperada | Reintentar; si persiste, revisar cuota de API en Google AI Studio |
| `RuntimeError` en MediaPipe | OpenCV no pudo abrir el video | Verificar que el archivo de video no esté corrupto y sea un formato compatible |
| El proceso tarda demasiado | Modelo Whisper grande en CPU | Usar `--whisper-model base` o `tiny`, o reducir la resolución del video |

---

## 11. Glosario

| Término | Definición |
|---|---|
| **DSS** | Decision Support System (Sistema de Apoyo a la Decisión) — herramienta de software que ayuda a los usuarios a tomar decisiones informadas proporcionando análisis de datos. |
| **MediaPipe** | Framework de Google para el procesamiento de video en tiempo real. RevFast lo usa para detectar rostro (478 landmarks) y pose corporal (33 landmarks). |
| **Whisper** | Modelo de reconocimiento automático del habla (ASR) desarrollado por OpenAI. Soporta múltiples idiomas y detecta el idioma automáticamente. |
| **Gemini** | Familia de modelos de lenguaje de Google. RevFast usa `gemini-1.5-pro` para interpretar la transcripción y los metadatos y generar una evaluación estructurada. |
| **CIL** | Centro Institucional de Lenguas de la UADY. Institución para la que fue desarrollado RevFast. |
| **UADY** | Universidad Autónoma de Yucatán. |
| **Rúbrica** | Instrumento de evaluación que define criterios y niveles de desempeño para calificar una actividad. |
| **Criterio** | Cada una de las dimensiones de evaluación dentro de una rúbrica (ej. vocabulario, fluidez). |
| **Clave** | Identificador único de un criterio en el JSON de rúbrica (ej. `vocabulario`, `precision_gramatical`). Debe seguir el formato `^[a-z][a-z0-9_]*$`. |
| **Umbral aprobatorio** | Puntuación total mínima (exclusiva) para que un alumno sea considerado aprobado. |

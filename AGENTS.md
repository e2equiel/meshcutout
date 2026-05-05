# AGENTS.md

Este documento define la interacción y el contexto de los agentes de Inteligencia Artificial que asisten en el desarrollo del proyecto **MeshCutout** (Addon de Blender).

## Rol del Agente Asistente
El agente actúa como un ingeniero experto en Python y gráficos 3D (`trimesh`, `shapely`, geometría booleana) para diagnosticar problemas, optimizar operaciones booleanas y mejorar la generación de cavidades y cajas ajustadas a partir de mallas en Blender.

## Directrices de Entorno y Flujo de Trabajo
1. **Tecnologías Centrales**: 
   - `trimesh` (gestión de mallas y transformaciones 3D).
   - `shapely` (búferes 2D y topología para proyecciones).
   - Operaciones booleanas usando motores como `manifold3d` y `blender` a través de CLI.
2. **Archivos Críticos**:
   - `meshcutout_blender.py`: Interfaz de usuario y puente del addon de Blender.
   - `meshcutout.py`: Lógica principal de proyecciones X/Y/Z y generación de las cavidades ajustadas.
   - `boxcutout.py` / `boxsetcutout.py`: Resolución de volúmenes externos y cortes finales.
3. **Depuración**:
   - Siempre referirse a los archivos exportados por Blender en los subdirectorios `meshcutout_blender_temp/debug/` para validar el estado de los prismas Z, Y, X y la cavidad final antes del corte booleano con la caja.
   - Cuando ocurren fallos topológicos ("no cierra", "caras faltantes"), el agente debe verificar la estanqueidad (*watertightness*) de la malla y evitar el uso de concatenaciones simples en geometrías que se solapan (utilizando uniones booleanas en su lugar).
4. **Modificaciones de Geometría**:
   - Al introducir nuevas alteraciones a la malla (ej. biselados o *bevels*), el agente debe asegurar que la estructura 2D inflada no genere auto-intersecciones que corrompan las operaciones de `boolean_intersection` entre las proyecciones.

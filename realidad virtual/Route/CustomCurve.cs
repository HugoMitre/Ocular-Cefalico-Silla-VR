using UnityEngine;
using System.Collections.Generic;

public class CustomCurve : MonoBehaviour
{
    public List<CurveAnchor> anchors = new List<CurveAnchor>();
    public LineRenderer pathRenderer;
    public int smoothness = 50;
    public float controlPointDefaultDistance = 2f;
    public float lineWidth = 0.2f; // Nuevo campo para el ancho
    public bool mostrarControles = false; // Para mantener el checkbox "Mostrar Controles"

    [System.Serializable]
    public class CurveAnchor
    {
        public Vector3 position;
        public Vector3 controlPointBack;
        public Vector3 controlPointForward;

        public CurveAnchor(Vector3 pos)
        {
            position = pos;
            controlPointBack = pos + (Vector3.left * 2f);
            controlPointForward = pos + (Vector3.right * 2f);
        }
    }

    void OnValidate()
    {
        // Se llama automáticamente cuando se cambia cualquier valor en el Inspector
        UpdateLineWidth();
        RedrawPath();
    }

    private void UpdateLineWidth()
    {
        if (pathRenderer != null)
        {
            pathRenderer.startWidth = lineWidth;
            pathRenderer.endWidth = lineWidth;
        }
    }

    public void RedrawPath()
    {
        if (pathRenderer == null || anchors.Count < 2) return;

        List<Vector3> curvePoints = new List<Vector3>();
        for (int i = 0; i < anchors.Count - 1; i++)
        {
            var currentAnchor = anchors[i];
            var nextAnchor = anchors[i + 1];

            for (float t = 0; t <= 1; t += 1f / smoothness)
            {
                curvePoints.Add(CalculateBezierPoint(
                    currentAnchor.position,
                    currentAnchor.controlPointForward,
                    nextAnchor.controlPointBack,
                    nextAnchor.position,
                    t
                ));
            }
        }

        pathRenderer.positionCount = curvePoints.Count;
        pathRenderer.SetPositions(curvePoints.ToArray());
        UpdateLineWidth(); // Asegura que el ancho se actualice
    }

    private Vector3 CalculateBezierPoint(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;
        float uuu = uu * u;
        float ttt = tt * t;

        return uuu * p0 + 3 * uu * t * p1 + 3 * u * tt * p2 + ttt * p3;
    }

    public void SuavizarPuntoControl(int anchorIndex)
    {
        if (anchorIndex < 0 || anchorIndex >= anchors.Count) return;

        var anchor = anchors[anchorIndex];
        float distanciaControl = controlPointDefaultDistance;

        Vector3 direccionPromedio = Vector3.zero;

        if (anchorIndex > 0)
            direccionPromedio += (anchor.position - anchors[anchorIndex - 1].position).normalized;
        if (anchorIndex < anchors.Count - 1)
            direccionPromedio += (anchors[anchorIndex + 1].position - anchor.position).normalized;

        if (direccionPromedio != Vector3.zero)
        {
            direccionPromedio.Normalize();

            float distanciaBack = Vector3.Distance(anchor.position, anchor.controlPointBack);
            float distanciaForward = Vector3.Distance(anchor.position, anchor.controlPointForward);

            anchor.controlPointBack = anchor.position - direccionPromedio * distanciaBack;
            anchor.controlPointForward = anchor.position + direccionPromedio * distanciaForward;
        }
    }

    public void SuavizarTodosLosPuntos()
    {
        for (int i = 0; i < anchors.Count; i++)
        {
            SuavizarPuntoControl(i);
        }
        RedrawPath();
    }

    public void AjustarEscalaPuntosControl(int anchorIndex, float escala)
    {
        if (anchorIndex < 0 || anchorIndex >= anchors.Count) return;

        var anchor = anchors[anchorIndex];
        Vector3 direccionBack = (anchor.controlPointBack - anchor.position).normalized;
        Vector3 direccionForward = (anchor.controlPointForward - anchor.position).normalized;
        float distanciaBase = controlPointDefaultDistance * escala;

        anchor.controlPointBack = anchor.position + direccionBack * distanciaBase;
        anchor.controlPointForward = anchor.position + direccionForward * distanciaBase;

        RedrawPath();
    }

    public Vector3 GetNearestPoint(Vector3 targetPosition)
    {
        if (pathRenderer.positionCount < 2) return transform.position;

        Vector3 closestPoint = Vector3.zero;
        float minDistance = float.MaxValue;

        for (int i = 0; i < pathRenderer.positionCount - 1; i++)
        {
            Vector3 startPoint = pathRenderer.GetPosition(i);
            Vector3 endPoint = pathRenderer.GetPosition(i + 1);
            Vector3 pointOnSegment = GetClosestPointOnSegment(startPoint, endPoint, targetPosition);
            float distance = Vector3.Distance(targetPosition, pointOnSegment);

            if (distance < minDistance)
            {
                minDistance = distance;
                closestPoint = pointOnSegment;
            }
        }

        return closestPoint;
    }

    private Vector3 GetClosestPointOnSegment(Vector3 segmentStart, Vector3 segmentEnd, Vector3 point)
    {
        Vector3 segment = segmentEnd - segmentStart;
        float segmentLength = segment.magnitude;
        Vector3 segmentDirection = segment.normalized;
        Vector3 pointToStart = point - segmentStart;
        float projection = Vector3.Dot(pointToStart, segmentDirection);
        projection = Mathf.Clamp(projection, 0f, segmentLength);

        return segmentStart + (segmentDirection * projection);
    }
}
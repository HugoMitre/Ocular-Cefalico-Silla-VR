using UnityEngine;

public class rotaciontecla : MonoBehaviour
{
    public float Speed = 5.0f;
    public float RotationSpeed = 100.0f;
    private Rigidbody rb;
    private float initialY;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }

        initialY = transform.position.y;

        rb.mass = 10f;
        rb.drag = 1f;
        rb.angularDrag = 1f;
        rb.freezeRotation = true;

        rb.constraints = RigidbodyConstraints.FreezeRotationX |
                        RigidbodyConstraints.FreezeRotationZ |
                        RigidbodyConstraints.FreezePositionY;

        rb.useGravity = false;
    }

    void FixedUpdate()
    {
        float rotation = 0f;
        Vector3 movement = Vector3.zero;

        // Rotación
        if (Input.GetKey(KeyCode.A))
        {
            rotation -= 1f;
        }
        if (Input.GetKey(KeyCode.D))
        {
            rotation += 1f;
        }

        // Movimiento - MODIFICADO AQUÍ
        if (Input.GetKey(KeyCode.W))
        {
            movement = -transform.right * Speed; // Movimiento hacia adelante
        }
        if (Input.GetKey(KeyCode.S))
        {
            movement = transform.right * Speed; // Movimiento hacia atrás
        }

        // Aplicar movimiento
        if (movement != Vector3.zero)
        {
            Vector3 newPosition = rb.position + movement * Time.fixedDeltaTime;
            newPosition.y = initialY;
            rb.MovePosition(newPosition);
        }

        // Aplicar rotación
        transform.Rotate(0, rotation * Time.fixedDeltaTime * RotationSpeed, 0);
    }
}
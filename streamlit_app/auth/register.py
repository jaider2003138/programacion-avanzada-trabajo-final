"""
Pantalla de registro de usuarios para Streamlit.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

try:
    from streamlit_app.components import render_panel_title, render_small_panel_title
except ModuleNotFoundError:
    from components import render_panel_title, render_small_panel_title


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")


def get_current_user() -> dict[str, str]:
    """
    Devuelve el usuario actual autenticado para enviarlo a la API.
    """
    return {
        "user_id": str(st.session_state.get("user_id") or ""),
        "user_name": str(st.session_state.get("user_name") or ""),
        "user_email": str(st.session_state.get("user_email") or ""),
        "user_role": str(st.session_state.get("user_role") or ""),
    }


def set_logged_user(user: dict[str, Any]) -> None:
    """
    Guarda el usuario autenticado en session_state.
    """
    st.session_state["user_id"] = user["id"]
    st.session_state["user_name"] = user["full_name"]
    st.session_state["user_email"] = user["email"]
    st.session_state["user_role"] = user["role"]


def bootstrap_admin(
    full_name: str,
    email: str,
    password: str,
) -> dict[str, Any] | None:
    """
    Crea el primer administrador cuando todavía no existen usuarios.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/users/bootstrap",
            json={
                "full_name": full_name,
                "email": email,
                "password": password,
            },
            timeout=20,
        )

        if response.status_code not in (200, 201):
            try:
                st.error(response.json().get("error", "No se pudo crear el administrador."))
            except Exception:
                st.error("No se pudo crear el administrador.")
            return None

        return response.json().get("user")

    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return None


def create_user_by_admin(
    full_name: str,
    email: str,
    password: str,
    role: str,
) -> dict[str, Any] | None:
    """
    Crea usuarios desde un usuario admin autenticado.
    """
    try:
        payload = {
            "full_name": full_name,
            "email": email,
            "password": password,
            "role": role,
            "status": "activo",
            **get_current_user(),
        }

        response = requests.post(
            f"{API_BASE_URL}/users",
            json=payload,
            timeout=20,
        )

        if response.status_code not in (200, 201):
            try:
                data = response.json()
                st.error(data.get("detail") or data.get("error") or "No se pudo crear el usuario.")
            except Exception:
                st.error("No se pudo crear el usuario.")
            return None

        return response.json().get("user")

    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API de usuarios: {exc}")
        return None


def _validate_register_form(
    full_name: str,
    email: str,
    password: str,
    confirm_password: str,
) -> bool:
    """
    Valida datos básicos del formulario.
    """
    if not full_name.strip():
        st.warning("El nombre completo es obligatorio.")
        return False

    if not email.strip():
        st.warning("El correo es obligatorio.")
        return False

    if "@" not in email:
        st.warning("Ingresa un correo válido.")
        return False

    if len(password) < 6:
        st.warning("La contraseña debe tener mínimo 6 caracteres.")
        return False

    if password != confirm_password:
        st.warning("Las contraseñas no coinciden.")
        return False

    return True


def show_register_page(mode: str = "admin_create") -> None:
    """
    Renderiza la pantalla de registro.

    mode='bootstrap': crea el primer admin.
    mode='admin_create': permite que un admin cree usuarios.
    """
    is_bootstrap = mode == "bootstrap"

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        with st.container(border=True):
            if is_bootstrap:
                render_panel_title("Crear primer administrador", "＋")
                st.caption(
                    "Usa esta opción solo la primera vez, cuando todavía no existen usuarios."
                )
            else:
                render_panel_title("Registrar usuario", "＋")
                st.caption(
                    "Crea usuarios para controlar quién manipula el inventario."
                )

            with st.form("register_form"):
                full_name = st.text_input(
                    "Nombre completo",
                    placeholder="Ejemplo: Jaider Gómez",
                )
                email = st.text_input(
                    "Correo electrónico",
                    placeholder="usuario@correo.com",
                )
                password = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="Mínimo 6 caracteres",
                )
                confirm_password = st.text_input(
                    "Confirmar contraseña",
                    type="password",
                    placeholder="Repite la contraseña",
                )

                if is_bootstrap:
                    role = "admin"
                    st.info("El primer usuario será creado con rol admin.")
                else:
                    role = st.selectbox(
                        "Rol del usuario",
                        options=["bodega", "admin"],
                        help="Bodega tiene permisos limitados. Admin puede administrar todo.",
                    )

                submitted = st.form_submit_button(
                    "Crear usuario",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if not _validate_register_form(
                    full_name,
                    email,
                    password,
                    confirm_password,
                ):
                    return

                if is_bootstrap:
                    user = bootstrap_admin(
                        full_name.strip(),
                        email.strip(),
                        password,
                    )

                    if user:
                        st.success("Administrador creado correctamente.")
                        set_logged_user(user)
                        st.session_state["auth_view"] = "login"
                        st.rerun()

                else:
                    user = create_user_by_admin(
                        full_name.strip(),
                        email.strip(),
                        password,
                        role,
                    )

                    if user:
                        st.success("Usuario creado correctamente.")

    with right_col:
        with st.container(border=True):
            render_small_panel_title("Clasificación de usuarios", "▣")

            st.markdown(
                """
                **Admin**
                
                - Ver inventario
                - Crear usuarios
                - Ver logs
                - Actualizar productos
                - Desactivar productos
                - Exportar reportes

                **Bodega**
                
                - Clasificar productos
                - Registrar productos
                - Hacer cargue masivo
                - Ver inventario
                - Consultar sus propios logs
                """
            )

            if is_bootstrap:
                if st.button("Volver al inicio de sesión", use_container_width=True):
                    st.session_state["auth_view"] = "login"
                    st.rerun()
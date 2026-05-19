"""
Pantalla de inicio de sesión para Streamlit.
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


def set_logged_user(user: dict[str, Any]) -> None:
    """
    Guarda el usuario autenticado en session_state.
    """
    st.session_state["user_id"] = user["id"]
    st.session_state["user_name"] = user["full_name"]
    st.session_state["user_email"] = user["email"]
    st.session_state["user_role"] = user["role"]


def login_user(email: str, password: str) -> dict[str, Any] | None:
    """
    Autentica usuario contra la API Flask.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={
                "email": email,
                "password": password,
            },
            timeout=15,
        )

        if response.status_code != 200:
            try:
                st.error(response.json().get("error", "No se pudo iniciar sesión."))
            except Exception:
                st.error("No se pudo iniciar sesión.")
            return None

        return response.json().get("user")

    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API de autenticación: {exc}")
        return None


def show_login_page() -> None:
    """
    Renderiza la pantalla de inicio de sesión.
    """
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        with st.container(border=True):
            render_panel_title("Iniciar sesión", "◉")
            st.caption(
                "Ingresa con tu correo y contraseña para acceder al sistema de inventario."
            )

            with st.form("login_form"):
                email = st.text_input(
                    "Correo electrónico",
                    placeholder="usuario@correo.com",
                )
                password = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="Ingresa tu contraseña",
                )

                submitted = st.form_submit_button(
                    "Entrar al sistema",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if not email.strip() or not password:
                    st.warning("Debes ingresar correo y contraseña.")
                else:
                    user = login_user(email.strip(), password)

                    if user:
                        set_logged_user(user)
                        st.success("Inicio de sesión correcto.")
                        st.rerun()

    with right_col:
        with st.container(border=True):
            render_small_panel_title("Roles del sistema", "▣")

            st.markdown(
                """
                **Admin**
                
                Puede administrar inventario, usuarios, logs y exportaciones.

                **Bodega**
                
                Puede clasificar productos, registrar inventario y consultar sus actividades.
                """
            )

            st.info(
                "Si todavía no existe ningún usuario, crea el primer administrador "
                "desde la pantalla de registro inicial."
            )

            if st.button("Crear primer administrador", use_container_width=True):
                st.session_state["auth_view"] = "register"
                st.session_state["register_mode"] = "bootstrap"
                st.rerun()
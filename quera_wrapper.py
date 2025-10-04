# quera_wrapper.py
import os
from enum import Enum
from typing import Optional
import logging

class QuEraWrapper:
    class Device(Enum):
        BLOQADE = "BLOQADE"
        AQUILA = "AQUILA"

    # ===== Class attributes =====
    program = None
    _device: "QuEraWrapper.Device" = Device.BLOQADE
    __zapier_webhook_url: Optional[str] = None
    __zapier_webhook_key: Optional[str] = None
    __vercel_api_url: Optional[str] = None
    # ===== Class configuration =====
    @classmethod
    def configure(
        cls,
        *,
        device: "QuEraWrapper.Device | str | None" = None,
        zapier_webhook_url: Optional[str] = None,
        zapier_webhook_key: Optional[str] = None,
        vercel_api_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        print("QuEraWrapper device: ",device)
        if device is not None:
            if isinstance(device, str):
                device = device.upper()
                if device == "AQUILA":
                    cls._device = cls.Device.AQUILA
            elif isinstance(device, cls.Device):
                cls._device = device
        cls._logger=logger
        # Credential store
        if zapier_webhook_url is not None:
            cls.__zapier_webhook_url = zapier_webhook_url
        if zapier_webhook_key is not None:
            cls.__zapier_webhook_key = zapier_webhook_key
        if vercel_api_url is not None:
            cls.__vercel_api_url = vercel_api_url


    # ===== Exec =====
    @classmethod
    def run(cls, **kwargs):
        if cls.program is None:
            raise RuntimeError("Program not found.")

        if cls._device == cls.Device.BLOQADE:
            # asume interfaz .bloqade.python().run(...)
            return cls.program.bloqade.python().run(**kwargs)

        # AQUILA: requiere creds
        missing = [
            name for name, val in dict(
                ZAPIER_WEBHOOK_URL=cls.__zapier_webhook_url,
                ZAPIER_WEBHOOK_KEY=cls.__zapier_webhook_key,
                VERCEL_API_URL=cls.__vercel_api_url,
            ).items() if not val
        ]
        if missing:
            raise RuntimeError(
                "Invalid credentials for AQUILA: " + ", ".join(missing)
            )

        # Set envs de forma temporal
        old_env = {}
        try:
            for k, v in {
                "ZAPIER_WEBHOOK_URL": cls.__zapier_webhook_url,
                "ZAPIER_WEBHOOK_KEY": cls.__zapier_webhook_key,
                "VERCEL_API_URL": cls.__vercel_api_url
            }.items():
                if v is not None:
                    old_env[k] = os.environ.get(k)
                    os.environ[k] = v  # type: ignore[arg-type]
            from bloqade.analog.task.exclusive import ExclusiveRemoteTask
            # asume interfaz .braket.aquila().run(...)
            return cls.program.quera.custom().run( RemoteTask=ExclusiveRemoteTask,**kwargs)
        finally:
            # Restaura envs
            for k, old in old_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
            
            

            
    
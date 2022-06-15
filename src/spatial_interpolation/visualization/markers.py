from typing import Dict, Tuple, Union
import folium

from functools import lru_cache


class IconMarker(folium.Marker):
    def __init__(
        self,
        location: Union[Tuple[float, float], str],
        popup: str = None,
        icon:str=None,
        prefix:str=None,
        icon_params: Dict = None,
        **kwargs,
    ): 
        icon_params = icon_params or {}
        icon_params["color"] = icon_params.get("color", kwargs.pop("color", "blue"))
        icon_params["prefix"] = icon_params.get("prefix", prefix)
        
        icon = self._get_icon(icon,**icon_params)
        super().__init__(location, popup=popup,icon=icon,**kwargs)
    

    def _get_icon(self,icon,**icon_params) -> folium.Icon:
        return folium.Icon(icon=icon,**icon_params)
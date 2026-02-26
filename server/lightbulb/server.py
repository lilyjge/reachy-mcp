"""Kasa Lightbulb MCP Server.

Run with:
    python -m server.lightbulb.server
"""

import argparse
from contextlib import asynccontextmanager
from fastmcp import FastMCP
from kasa import Discover

# Global device instance
_device = None

# Credentials
LIGHTBULB_IP = "10.120.2.79"  # Replace with your lightbulb's IP address
LIGHTBULB_USERNAME = "" # Replace with your email for Tapo 
LIGHTBULB_PASSWORD = "" # Replace with password


@asynccontextmanager
async def lifespan(server):
    """Initialize and cleanup the lightbulb connection."""
    global _device
    
    # Discover and connect to the lightbulb
    _device = await Discover.discover_single(
        LIGHTBULB_IP,
        username=LIGHTBULB_USERNAME,
        password=LIGHTBULB_PASSWORD
    )
    await _device.update()
    print(f"Connected to {_device.alias}")
    
    try:
        yield
    finally:
        # Cleanup
        if _device:
            await _device.protocol.close()
        print("Lightbulb connection closed")


def register_lightbulb_tools(mcp: FastMCP):
    """Register all lightbulb control tools."""

    @mcp.tool()
    async def turn_on_light() -> str:
        """Turn on the lightbulb."""
        await _device.turn_on()
        await _device.update()
        return f"Light turned on. Current state: {_device.is_on}"

    @mcp.tool()
    async def turn_off_light() -> str:
        """Turn off the lightbulb."""
        await _device.turn_off()
        await _device.update()
        return f"Light turned off. Current state: {_device.is_on}"

    @mcp.tool()
    async def toggle_light() -> str:
        """Toggle the lightbulb on/off."""
        if _device.is_on:
            await _device.turn_off()
        else:
            await _device.turn_on()
        await _device.update()
        return f"Light toggled. Current state: {'ON' if _device.is_on else 'OFF'}"

    @mcp.tool()
    async def set_brightness(brightness: int) -> str:
        """
        Set the brightness of the lightbulb.
        
        Args:
            brightness: Brightness level (0-100)
        """
        if not 0 <= brightness <= 100:
            return "Error: Brightness must be between 0 and 100"
        
        await _device.modules["Light"].set_brightness(brightness)
        await _device.update()
        return f"Brightness set to {brightness}%"

    @mcp.tool()
    async def set_color_hsv(hue: int, saturation: int, brightness: int) -> str:
        """
        Set the color of the lightbulb using HSV values.
        
        Args:
            hue: Hue value (0-360)
            saturation: Saturation percentage (0-100)
            brightness: Brightness percentage (0-100)
        """
        if not 0 <= hue <= 360:
            return "Error: Hue must be between 0 and 360"
        if not 0 <= saturation <= 100:
            return "Error: Saturation must be between 0 and 100"
        if not 0 <= brightness <= 100:
            return "Error: Brightness must be between 0 and 100"
        
        await _device.modules["Light"].set_hsv(hue, saturation, brightness)
        await _device.update()
        return f"Color set to HSV({hue}, {saturation}, {brightness})"

    @mcp.tool()
    async def set_color_temperature(temp: int) -> str:
        """
        Set the color temperature of the lightbulb.
        
        Args:
            temp: Color temperature in Kelvin (2500-6500)
        """
        if not 2500 <= temp <= 6500:
            return "Error: Color temperature must be between 2500K and 6500K"
        
        await _device.modules["Light"].set_color_temp(temp)
        await _device.update()
        return f"Color temperature set to {temp}K"

    @mcp.tool()
    async def set_preset_color(color: str) -> str:
        """
        Set the lightbulb to a preset color.
        
        Args:
            color: Color name (red, green, blue, purple, yellow, orange, pink, white, warm_white, cool_white)
        """
        color_map = {
            "red": (0, 100, 100),
            "green": (120, 100, 100),
            "blue": (240, 100, 100),
            "purple": (280, 100, 100),
            "yellow": (60, 100, 100),
            "orange": (30, 100, 100),
            "pink": (330, 100, 100),
            "white": (0, 0, 100),
            "warm_white": (30, 20, 100),
            "cool_white": (200, 20, 100),
        }
        
        if color.lower() not in color_map:
            return f"Error: Unknown color. Available: {', '.join(color_map.keys())}"
        
        h, s, v = color_map[color.lower()]
        await _device.modules["Light"].set_hsv(h, s, v)
        await _device.update()
        return f"Color set to {color}"

    @mcp.tool()
    async def get_light_status() -> str:
        """Get the current status of the lightbulb."""
        await _device.update()
        light = _device.modules.get("Light")
        
        status = f"""
Lightbulb Status:
- Device: {_device.alias}
- State: {'ON' if _device.is_on else 'OFF'}
- Brightness: {light.brightness}%
- HSV: {light.hsv}
- Color Temperature: {light.color_temp}K
- RSSI: {_device.rssi} dBm
- SSID: {_device.modules.get('DeviceModule', {}).ssid if hasattr(_device.modules.get('DeviceModule', {}), 'ssid') else 'N/A'}
"""
        return status.strip()


def main():
    """Main entry point for the lightbulb MCP server."""
    parser = argparse.ArgumentParser(description="Kasa Lightbulb MCP Server")
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port to run the server on (default: 5002)",
    )
    args = parser.parse_args()

    mcp = FastMCP("Kasa Lightbulb Controller", lifespan=lifespan)
    register_lightbulb_tools(mcp)
    mcp.run(transport="streamable-http", port=args.port, stateless_http=True)


if __name__ == "__main__":
    main()

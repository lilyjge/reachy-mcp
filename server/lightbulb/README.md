# Kasa Lightbulb MCP Server

MCP server for controlling a Kasa/Tapo smart lightbulb.

## Setup

The lightbulb credentials are configured in [server.py](server.py). Please fill in your details:

- IP: `10.120.6.140`
- Username: ``
- Password: ``

## Running the Server

Start the server on port 5002:

```bash
python -m server.lightbulb.server --port 5002
```

## Available Tools

### Basic Controls

- **turn_on_light()** - Turn on the lightbulb
- **turn_off_light()** - Turn off the lightbulb
- **toggle_light()** - Toggle the lightbulb on/off

### Brightness

- **set_brightness(brightness: int)** - Set brightness (0-100)

### Color Control

- **set_color_hsv(hue: int, saturation: int, brightness: int)**
  - hue: 0-360
  - saturation: 0-100
  - brightness: 0-100

- **set_preset_color(color: str)** - Set to a preset color
  - Available colors: red, green, blue, purple, yellow, orange, pink, white, warm_white, cool_white

- **set_color_temperature(temp: int)** - Set color temperature (2500-6500K)

### Status

- **get_light_status()** - Get current lightbulb status

## Testing

Run the test script:

```bash
python test.py  # Simple on/off/color test
```

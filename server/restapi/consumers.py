from channels.generic.websocket import AsyncJsonWebsocketConsumer
from restapi.utils import get_scene, save_scene, get_user, save_upload, get_latest_upload, mark_as_inserted


class BidderConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        print("Connected")
        await self.accept()
        # Add to group
        await self.channel_layer.group_add("bidding", self.channel_name)
        # Add channel to group
        await self.send_json({"msg_type": "connected"})

    async def receive_json(self, content, **kwargs):
        try:
            price = int(content.get("price"))
            scene_id = int(content.get("item_id"))
        except:
            await self.send_json({"error": "invalid argument"})
            return

        print("receive price ", price)
        print("receive item_id ", scene_id)

        scene = await get_scene(scene_id)

        # Check if the last bidder is the same as the current bidder
        if scene.inserted_ad and scene.inserted_ad.owner.username == self.scope["user"].username:
            await self.send_json({"error": "The same user cannot bid twice!"})

        # Update bidding price
        if price > scene.highest_price:
            scene.highest_price = price
            # Get the latest upload from the user
            user = await get_user(self.scope["user"].username)
            latest_upload = await get_latest_upload(user)
            # Mark last bidder's upload not inserted
            if scene.inserted_ad:
                scene.inserted_ad.inserted = False
                await save_upload(scene.inserted_ad)

            scene.inserted_ad = latest_upload
            await save_scene(scene)
            await mark_as_inserted(latest_upload)
            # Broadcast new bidding price
            print("performing group send")
            await self.channel_layer.group_send(
                "bidding",
                {
                    "type": "update.price",
                    "price": price,
                    "scene_id": scene_id,
                }
            )
        else:
            await self.send_json({"error": "Bidding price lower than current price"})

    async def update_price(self, event):
        print("sending new price")
        await self.send_json({
            "msg_type": "update",
            "scene_id": event["scene_id"],
            "price": event["price"],
        })

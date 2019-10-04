from django.urls import path
from . import views


app_name = "restapi"

# video list and create
video_list = views.VideoViewSet.as_view({
    "get": "list",
    "post": "create",
})

# video retrieve, destroy and update
video_detail = views.VideoViewSet.as_view({
    "get": "retrieve",
    "delete": "destroy",
    "patch": "partial_update"
})

# user list and create
user_list = views.UserViewSet.as_view({
    "get": "list",
    "post": "create",
})

# user retrieve, destroy and update
user_detail = views.UserViewSet.as_view({
    "get": "retrieve",
    "delete": "destroy",
    "patch": "partial_update"
})

# dlmodel list and create
dlmodel_list = views.DlModelViewSet.as_view({
    "get": "list",
    "post": "create",
})

# dlmodel retrieve, destroy and update
dlmodel_detail = views.DlModelViewSet.as_view({
    "get": "retrieve",
    "delete": "destroy",
    "patch": "partial_update"
})

default_dlmodel_list = views.DlModelViewSet.as_view({
    "get": "get_default_dlmodels",
})

# For react client
urlpatterns = [
    path('logout/', views.Logout.as_view(), name="user-logout"),
    path('videos/', video_list, name="video-list"),
    path('videos/<int:pk>/', video_detail, name="video-detail"),
    path('users/', user_list, name="user-list"),
    path('users/<int:pk>/', user_detail, name="user-detail"),
    path('dlmodels/', dlmodel_list, name="dlmodel-list"),
    path('dlmodels/<int:pk>/', dlmodel_detail, name="dlmodel-detail"),
    path('defaultdlmodels/', default_dlmodel_list, name="default-dlmodel-list"),
    path('scenesearch/', views.SceneSearch.as_view(), name="scene-search"),
    path('productinsert/', views.ProductInsert.as_view(), name="product-insert"),
]

# For android client
urlpatterns += [
    path('getdemo/', views.GetDemoVideoUrl.as_view(), name="get-demo"),
    path('search/', views.SearchProduct.as_view(), name="search-product"),
    path('getmyvideos/', views.GetUserVideoUrls.as_view(), name="get-my-videos"),
]

# For dashboard table retrieval
urlpatterns += [
    path('retrievetable/<int:pk>/', views.RetrieveTable.as_view(), name="table-retrieve"),
]
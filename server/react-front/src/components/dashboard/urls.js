import React from 'react';
import DashboardApp from "./index";
import VideoRouter from "./video/urls";
import UserRouter from "./user/urls";
import Route from "../../routes/RouteWrapper";
import Switch from "react-router/Switch";
import WatchApp from "./watch"
import ModelRouter from "./model/urls";
import AdRouter from "./productSearch/urls";

const DashboardRouter = () => (
    <Switch>
        {Route({exact: true, path: '/', component: DashboardApp})}
        {Route({path: '/video', component: VideoRouter})}
        {Route({path: '/user', component: UserRouter})}
        {Route({path: '/model', component: ModelRouter})}
        {Route({path: '/watch', component: WatchApp})}
        {Route({path: '/ad', component: AdRouter})}
        {/*<Redirect to={'404'}/>*/}
    </Switch>
);

export default DashboardRouter;

import React from 'react';
import Switch from 'react-router-dom/Switch';
import ReactRouter from "react-router-dom/Router";
import createBrowserHistory from 'history/createBrowserHistory';
import RouteWrapper from "./routes/RouteWrapper";
import PageNotFound from "./components/common/app/PageNotFound";
import DashboardRouter from "./components/dashboard/urls";
import HomeApp from "./components/home/index";
import withStyles from "@material-ui/core/styles/withStyles";
import Header from "./components/common/app/Header";

const history = createBrowserHistory();

const classes = theme => ({
    root: {
        display: 'flex',
        flexDirection: "column",
        minHeight: "100%",
    },
    content: {
        flexGrow: 1,
        backgroundColor: theme.palette.background.default,
        padding: theme.spacing.unit * 3,
        minWidth: 0,
        minHeight: "100%",
    },
    toolbar: theme.mixins.toolbar,
});

const RootRouter = ({classes}) => (
    <ReactRouter history={history}>
        <div className={classes.root}>
            <Header/>
            <main className={classes.content}>
                <div className={classes.toolbar}/>
                <Switch>
                    {RouteWrapper({path: '/', exact: true, component: HomeApp})}
                    {RouteWrapper({path: '/dashboard/', component: DashboardRouter})}
                    {RouteWrapper({component: PageNotFound})}
                </Switch>
            </main>
            <div id="hysia-footer-container"/>
        </div>
    </ReactRouter>
);

export default withStyles(classes)(RootRouter);

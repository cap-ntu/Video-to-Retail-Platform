import React from 'react';
import * as ReactDom from "react-dom";
import * as PropTypes from "prop-types";
import HysiaTitle from "./HysiaTitle";
import withStyles from "@material-ui/core/styles/withStyles";
import Footer from "../common/app/Footer";

const styles = () => ({
    root: {
        flexGrow: 1,
    },
});

const HomeApp = ({classes}) => {
    const footer = document.getElementById("hysia-footer-container");
    if (footer)
        ReactDom.createPortal(<Footer/>, footer);

    return (
        <div className={classes.root}>
            <HysiaTitle/>
        </div>
    );
};

HomeApp.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(HomeApp);

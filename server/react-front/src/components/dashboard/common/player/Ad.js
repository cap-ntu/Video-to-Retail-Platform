import React from "react";
import * as PropTypes from "prop-types";
import Grow from "@material-ui/core/es/Grow/Grow";
import withStyles from "@material-ui/core/styles/withStyles";
import IconButton from "@material-ui/core/es/IconButton/IconButton";
import CloseIcon from "@material-ui/icons/CloseRounded";
import {isEmpty} from "../../../../utils/utils";

const styles = theme => ({
    root: {
        width: 728,
        maxWidth: "100%",
        height: 90,
        position: "absolute",
        bottom: 45,
        left: "50%",
    },
    content: {
        width: "100%",
        height: "100%",
        backgroundColor: theme.palette.grey[50],
        backgroundSize: "contain",
        marginLeft: "-50%",
        cursor: "pointer",
    },
    buttonContainer: {
        backgroundColor: theme.palette.grey[50],
        opacity: 0.9,
    },
    buttonRoot: {
        padding: 6,
    },
});

class Ad extends React.PureComponent {

    state = {
        on: false,
    };

    componentWillUpdate(nextProps, nextState, nextContext) {
        this.setState({on: !isEmpty(nextProps.ad)});
    }

    handleOpenAd = () => {
        window.open(this.props.ad.url, "_black");
        window.focus();
    };

    render() {
        const {classes, ad} = this.props;
        const {on} = this.state;

        return (
            <Grow in={on}>
                <div className={classes.root}>
                    <div className={classes.content}
                         style={{backgroundImage: `url(${ad.poster || require("../../../../asset/hysia-ads.png")})`}}
                         onClick={this.handleOpenAd}>
                        <div style={{display: "flex"}}>
                            <div style={{flexGrow: 1}}/>
                            <div className={classes.buttonContainer}>
                                <IconButton classes={{root: classes.buttonRoot}}
                                            onClick={() => this.setState({on: false})}>
                                    <CloseIcon fontSize="small"/>
                                </IconButton>
                            </div>
                        </div>
                    </div>
                </div>
            </Grow>
        );
    }
}

Ad.defaultProps = {
    ad: {}
};

Ad.propTypes = {
    classes: PropTypes.object.isRequired,
    ad: PropTypes.shape({
        poster: PropTypes.string,
        url: PropTypes.string,
    }),
};

export default withStyles(styles)(Ad);

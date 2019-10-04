import React from "react";
import * as ReactDom from "react-dom";
import * as PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import DPlayer from "dplayer";
import Typography from "@material-ui/core/Typography";
import DetectionBox from "./DetectionBox";
import Ad from "./Ad";
import {isEmpty} from "../../../../utils/utils";

let secondaryMain;

const styles = theme => {
    // register primary color
    secondaryMain = theme.palette.secondary.main;

    return ({
        root: {
            position: "relative",
        },
        overlay: {
            position: "absolute",
            width: "100%",
            height: "100%",
            bottom: 0,
        },
    });
};

class Player extends React.PureComponent {

    state = {
        load: false,
        frame: 0,
        time: 0,
    };

    selfDom = null;

    componentDidMount() {
        this._interval = setInterval(this.handleUpdateFrame, 200);
        this._interval2 = setInterval(this.handleUpdateTime, 200);
        this.selfDom = ReactDom.findDOMNode(this);
        window.onresize = this.handleResize;
    }

    componentWillUnmount() {
        clearInterval(this._interval);
        clearInterval(this._interval2);
    }

    componentWillUpdate(nextProps, nextState, nextContext) {

        const {video} = nextProps;

        if (isEmpty(this.props.video) && !isEmpty(video)) {

            this.dp = new DPlayer({
                container: document.getElementById('dplayer-container'),
                theme: secondaryMain,
                video: video,
            });

            this.dp.on("loadeddata", () => {

                // create detection box root
                const detectionBoxRoot = document.createElement("div");
                detectionBoxRoot.setAttribute("id", "hysia-detectionBox-container");
                detectionBoxRoot.setAttribute("class", this.props.classes.overlay);
                document.querySelector("#dplayer-container > div.dplayer-video-wrap").insertBefore(
                    detectionBoxRoot,
                    document.querySelector("div.dplayer-danmaku")
                );

                this.setState({load: true});
                this.handleResize();
            });

            this.dp.on("play", () => this.handleUpdateEvent({play: true, progress: this.getProgress()}));
            this.dp.on("pause", () => this.handleUpdateEvent({play: false, progress: this.getProgress()}));
            this.dp.on("seeking", () => this.handleUpdateEvent({play: false, progress: this.getProgress()}));
            this.dp.on("seeked", () => this.handleUpdateEvent({play: !this.dp.video.paused}));
        }
    }

    getProgress = () =>
        this.state.load ? this.dp.video.currentTime / this.dp.video.duration : 0;

    // TODO: to be remove after the change of result api
    handleUpdateFrame = () => {
        const {handleUpdateParentState, video} = this.props;
        const frame = this.state.load ?
            Math.floor(this.getProgress() * (video.totalFrame || 1)) : 0;
        this.setState({frame});
        handleUpdateParentState({frame});
    };

    handleUpdateTime = () => {
        const {handleUpdateParentState} = this.props;
        const time = this.state.load ? this.dp.video.currentTime : 0;
        this.setState({time});
        handleUpdateParentState({time});
    };

    handleUpdateEvent = state => {
        this.props.handleUpdateParentState(_state => ({..._state, events: {..._state.events, ...state}}));
    };

    handleResize = () => {
        this.props.handleUpdateParentState({height: this.selfDom ? this.selfDom.clientHeight : 0});
    };

    render() {
        const {classes, boxOn, boxes, ads} = this.props;
        const {load} = this.state;

        return (
            <div className={classes.root}>
                <Typography component="div" id="dplayer-container"/>
                {
                    load ?
                        <React.Fragment>
                            <DetectionBox on={boxOn} boxes={boxes}/>
                            <Ad ad={ads}/>
                        </React.Fragment> :
                        null
                }
            </div>);
    }
}

Player.propTypes = {
    classes: PropTypes.object.isRequired,
    boxOn: PropTypes.bool.isRequired,
    boxes: PropTypes.arrayOf(
        PropTypes.shape({
            left: PropTypes.string.isRequired,
            right: PropTypes.string.isRequired,
            top: PropTypes.string.isRequired,
            bottom: PropTypes.string.isRequired,
        }).isRequired,
    ),
    video: PropTypes.shape({
        url: PropTypes.string,
        poster: PropTypes.string,
        description: PropTypes.string,
        totalFrame: PropTypes.number,
    }).isRequired,
    products: PropTypes.shape({
        poster: PropTypes.string.isRequired,
        url: PropTypes.string.isRequired,
        start: PropTypes.number.isRequired,
        end: PropTypes.number.isRequired,
    }),
    handleUpdateParentState: PropTypes.func,
};

export default withStyles(styles)(Player);

import React from "react";
import * as PropTypes from "prop-types";
import CardContent from "@material-ui/core/CardContent";
import WaveSurfer from "wavesurfer.js"
import Grow from "@material-ui/core/Grow";
import Collapse from "@material-ui/core/Collapse";
import classNames from "classnames";
import withStyles from "@material-ui/core/styles/withStyles";
import {emptyFunction} from "../../../utils/utils";

let primaryColor = null;
let secondaryColor = null;
const styles = theme => {
    secondaryColor = theme.palette.secondary.main;
    primaryColor = theme.palette.primary.main;
    return {
        root: {
            pointerEvents: "none",
        }
    }
};

class AudioCard extends React.PureComponent {

    state = {
        load: false,
    };

    componentDidMount() {
        this.wavesurfer = WaveSurfer.create({
            container: "#hysia-audioCard-root",
            waveColor: secondaryColor,
            progressColor: primaryColor,
            barWidth: 5,
            normalize: true,
            scrollParent: true,
            hideScrollbar: true,
        });

        this.wavesurfer.on("ready", () => {
            this.props.onReady();
            this.setState({load: true});
            this.handleSync(this.props.events);
            this.wavesurfer.setMute(true);
        });
        this.wavesurfer.on("destroy", () => this.setState({load: false}));
    }

    componentWillUpdate(nextProps, nextState, nextContext) {
        const {audio, events} = nextProps;

        if (audio && !this.props.audio)
            this.wavesurfer.load(audio);

        this.handleSync(events);
    }

    handleSync(events) {
        const {play, progress} = events;
        if (this.state.load) {
            this.wavesurfer.seekAndCenter(progress);
            play ? this.wavesurfer.play() : this.wavesurfer.pause();
        }
    }

    render() {
        const {classes, clickable, on} = this.props;

        return (
            <Collapse in={on}>
                <Grow in={on}>
                    <CardContent>
                        <div className={classNames({[classes.root]: !clickable})} id="hysia-audioCard-root"/>
                    </CardContent>
                </Grow>
            </Collapse>
        );
    }
}

AudioCard.defaultProps = {
    on: true,
    onReady: emptyFunction,
    clickable: false,
};

AudioCard.propTypes = {
    classes: PropTypes.object.isRequired,
    audio: PropTypes.string,
    on: PropTypes.bool,
    events: PropTypes.object.isRequired,
    onReady: PropTypes.func,
    clickable: PropTypes.bool,
};

export default withStyles(styles)(AudioCard);
